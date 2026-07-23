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
    src_draws = nothing
    n_draws = 1
    for a in bc.args
        if a isa RandomDraw
            if src_draws === nothing
                src_draws = a.draws
            end
            # Draw axis must hold the full broadcast result; a 1-draw constant recycles.
            n_draws = max(n_draws, size(a.draws, 1))
        end
    end
    axs = axes(bc)
    if src_draws === nothing
        return Array{T}(undef, axs...)
    end
    nc = _combine_nchains(bc.args...)
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

# Extract the scalar RV at broadcast position `idx` (a CartesianIndex over the
# destination shape). Singleton dims of this argument are held at index 1 so ordinary
# broadcast expansion works; the flattened column matches column-major LinearIndices.
function _bcast_elem(x::RandomDraw{T}, idx::CartesianIndex) where {T}
    flat = reshape(draws(x), size(draws(x), 1), :)
    sz = size(x)
    if isempty(sz)
        data = flat[:, 1]
        return RandomDraw{T, 0, typeof(data)}(data, x.nchains)
    end
    proj = ntuple(k -> sz[k] == 1 ? 1 : idx[k], length(sz))
    col = LinearIndices(sz)[proj...]
    data = flat[:, col]
    RandomDraw{T, 0, typeof(data)}(data, x.nchains)
end
