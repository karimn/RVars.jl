struct RandomDraw{T, N, A <: AbstractArray{T}} <: AbstractArray{T, N}
    draws::A
    nchains::Int

    function RandomDraw{T, N, A}(draws::A, nchains::Int=1) where {T, N, A <: AbstractArray{T}}
        nchains >= 1 || error("nchains must be >= 1")
        nd = size(draws, 1)
        if nd != 1 && nd % nchains != 0
            error("Number of chains ($nchains) does not divide number of draws ($nd)")
        end
        expected_dims = N + 1
        actual_dims = ndims(draws)
        if actual_dims != expected_dims
            error("Expected draws array with $expected_dims dimensions (draws × shape), got $actual_dims")
        end
        new{T, N, A}(draws, nchains)
    end
end

Base.eltype(::Type{<:RandomDraw{T}}) where {T} = T
Base.ndims(::Type{<:RandomDraw{T, N}}) where {T, N} = N

function Base.size(x::RandomDraw)
    n_extra = ndims(x.draws) - 1
    ntuple(i -> size(x.draws, i + 1), n_extra)
end

function Base.size(x::RandomDraw, d::Int)
    nd = ndims(x.draws) - 1
    if d <= nd
        return size(x.draws, d + 1)
    end
    return 1
end

Base.length(x::RandomDraw) = prod(size(x))

function Base.axes(x::RandomDraw)
    n_extra = ndims(x.draws) - 1
    ntuple(i -> axes(x.draws, i + 1), n_extra)
end

function Base.axes(x::RandomDraw, d::Int)
    nd = ndims(x.draws) - 1
    if d <= nd
        return axes(x.draws, d + 1)
    end
    return Base.OneTo(1)
end

function _broadcast_draws(x::RandomDraw, n::Int)
    old = draws(x)
    sz = size(old)
    if sz[1] == n
        return old
    end
    if sz[1] != 1
        error("Cannot broadcast draws: $(sz[1]) to $n")
    end
    new_sz = (n, sz[2:end]...)
    new_data = similar(old, new_sz)
    ci = CartesianIndices(sz[2:end])
    for i in 1:n
        for c in ci
            new_data[i, c] = old[1, c]
        end
    end
    return new_data
end

# Combine nchains across operands of an elementwise/broadcast operation. An operand
# with a single draw is a constant with no chain structure, so it defers to the other
# operand's nchains rather than forcing a collapse. Two genuine but differing chain
# counts cannot be aligned, so the result loses its chain structure (nchains = 1).
function _combine_nchains(operands...)
    nc = 0  # 0 = not yet seen a non-constant operand
    for x in operands
        x isa RandomDraw || continue
        size(x.draws, 1) == 1 && continue  # constant: chain-agnostic
        c = x.nchains
        if nc == 0
            nc = c
        elseif c != nc
            nc = 1
        end
    end
    return nc == 0 ? 1 : nc
end
