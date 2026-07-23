Base.broadcastable(x::RandomDraw) = x

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

function _max_ndraws(bc)
    max_nd = 1
    for a in bc.args
        if a isa RandomDraw
            nd = ndraws(a)
            nd > max_nd && (max_nd = nd)
        end
    end
    max_nd
end

function _pad_draws(d, target_nd)
    nd = size(d, 1)
    nd == target_nd && return d
    ndims_d = ndims(d)
    factor = cld(target_nd, nd)
    inner = (factor, ntuple(i -> 1, max(0, ndims_d - 1))...)
    result = repeat(d; inner=inner)
    if ndims_d == 1
        return result[1:target_nd]
    end
    result[1:target_nd, ntuple(i -> Colon(), ndims_d - 1)...]
end

function Base.similar(bc::Base.Broadcast.Broadcasted{RandomDrawStyle{N}}, ::Type{T}) where {N, T}
    max_nc = 1
    src_draws = nothing
    for a in bc.args
        if a isa RandomDraw
            nd = ndraws(a)
            nc = nchains(a)
            if nc > max_nc
                max_nc = nc
            end
            if src_draws === nothing || nd > size(src_draws, 1)
                src_draws = a.draws
            end
        end
    end
    axs = axes(bc)
    if src_draws === nothing
        return Array{T}(undef, axs...)
    end
    max_nd = size(src_draws, 1)
    sz = (max_nd, length.(axs)...)
    data = similar(src_draws, T, sz)
    RandomDraw{T, N, typeof(data)}(data, max_nc)
end

function Base.copy(bc::Base.Broadcast.Broadcasted{RandomDrawStyle{N}}) where {N}
    sample_args = map(a -> a isa AbstractArray ? a[1] : a, bc.args)
    sample_result = bc.f(sample_args...)
    T_rd = typeof(sample_result)
    T_inner = eltype(T_rd)
    dest = similar(bc, T_inner)
    max_nd = size(dest.draws, 1)
    prepared = map(bc.args) do a
        if a isa RandomDraw
            d = _pad_draws(draws(a), max_nd)
            nd = ndims(d)
            nd == 1 ? reshape(d, size(d, 1), 1) : d
        elseif a isa AbstractArray
            reshape(a, 1, size(a)...)
        else
            a
        end
    end
    dest.draws .= bc.f.(prepared...)
    dest
end