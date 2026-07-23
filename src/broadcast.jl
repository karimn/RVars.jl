Base.broadcastable(x::RandomDraw) = x

# Custom broadcast style to ensure output is RandomDraw
struct RandomDrawStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end

function Base.BroadcastStyle(::Type{<:RandomDraw{T, N}}) where {T, N}
    RandomDrawStyle{N}()
end

function Base.BroadcastStyle(s1::RandomDrawStyle{M}, s2::RandomDrawStyle{N}) where {M, N}
    RandomDrawStyle{max(M, N)}()
end

function Base.BroadcastStyle(s::RandomDrawStyle{M}, ::Base.Broadcast.DefaultArrayStyle{N}) where {M, N}
    RandomDrawStyle{max(M, N)}()
end

function Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{M}, s::RandomDrawStyle{N}) where {M, N}
    RandomDrawStyle{max(M, N)}()
end

function Base.BroadcastStyle(s::RandomDrawStyle{M}, ::Base.Broadcast.Style{Tuple}) where {M}
    RandomDrawStyle{M}()
end

function Base.similar(bc::Base.Broadcast.Broadcasted{RandomDrawStyle{N}}, ::Type{T}) where {N, T}
    nc = 1
    src_draws = nothing
    for a in bc.args
        if a isa RandomDraw
            nc = a.nchains
            if src_draws === nothing
                src_draws = a.draws
            end
        end
    end
    axs = axes(bc)
    if src_draws === nothing
        return Array{T}(undef, axs...)
    end
    n_draws = size(src_draws, 1)
    sz = (n_draws, length.(axs)...)
    data = similar(src_draws, T, sz)
    RandomDraw{T, N, typeof(data)}(data, nc)
end

function Base.copy(bc::Base.Broadcast.Broadcasted{RandomDrawStyle{N}}) where {N}
    sample_args = map(a -> a isa RandomDraw ? a[1] : a, bc.args)
    sample_result = bc.f(sample_args...)
    T_rd = typeof(sample_result)
    T_inner = eltype(T_rd)
    dest = similar(bc, T_inner)
    for I in CartesianIndices(size(dest))
        args = map(a -> a isa RandomDraw ? _bcast_elem(a, I) : a, bc.args)
        val = bc.f(args...)
        dest[I] = val
    end
    return dest
end

function _bcast_elem(x::RandomDraw{T}, idx::CartesianIndex) where {T}
    d = draws(x)
    sz = size(d)
    n_draws = sz[1]
    rest_sz = sz[2:end]
    n_rest = prod(rest_sz)
    if n_rest == 1
        return RandomDraw{T, 0, typeof(d[:, 1])}(d[:, 1], x.nchains)
    end
    linear_idx = 1
    for i in 1:length(rest_sz)
        stride = prod(rest_sz[i+1:end])
        di = idx[i]
        if di < 1 || di > rest_sz[i]
            throw(BoundsError(x, idx))
        end
        linear_idx += (di - 1) * stride
    end
    if linear_idx < 1 || linear_idx > n_rest
        throw(BoundsError(x, idx))
    end
    data = d[:, linear_idx]
    RandomDraw{T, 0, typeof(data)}(data, x.nchains)
end
