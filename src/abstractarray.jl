Base.IndexStyle(::Type{<:RandomDraw}) = IndexLinear()

# The store has the draws on axis 1 and the visible shape on axes 2..N+1. Collapse
# those trailing axes into a single (ndraws, nelements) matrix so an element's draws
# are a valid `[:, col]` slice. Column j here matches column-major linear index j of
# the visible array (and `LinearIndices(size(x))`), so linear/Cartesian access agree.
_flat_store(x::RandomDraw) = reshape(x.draws, size(x.draws, 1), :)

function Base.getindex(x::RandomDraw{T, N}, i::Int) where {T, N}
    @boundscheck checkbounds(x, i)
    data = _flat_store(x)[:, i]
    RandomDraw{T, 0, typeof(data)}(data, x.nchains)
end

function Base.getindex(x::RandomDraw{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(x, I...)
    col = LinearIndices(size(x))[I...]
    data = _flat_store(x)[:, col]
    RandomDraw{T, 0, typeof(data)}(data, x.nchains)
end

function Base.getindex(x::RandomDraw{T, N}, idx::AbstractArray{Bool}) where {T, N}
    if length(idx) != length(x)
        throw(DimensionMismatch("logical index length $(length(idx)) != length $(length(x))"))
    end
    result_draws = _flat_store(x)[:, vec(idx)]
    RandomDraw{eltype(result_draws), 1, typeof(result_draws)}(result_draws, x.nchains)
end

function Base.setindex!(x::RandomDraw{T, N}, val::Number, i::Int) where {T, N}
    @boundscheck checkbounds(x, i)
    _flat_store(x)[:, i] .= val
    return x
end

function Base.setindex!(x::RandomDraw, val::RandomDraw, i::Int)
    @boundscheck checkbounds(x, i)
    vd = draws(val)
    if size(vd, 1) == 1
        vd = repeat(vd, size(x.draws, 1))
    end
    _flat_store(x)[:, i] .= vec(vd)
    return x
end

function Base.reshape(x::RandomDraw{T, N}, dims::Dims) where {T, N}
    new_dims = (size(x.draws, 1), dims...)
    reshaped = reshape(x.draws, new_dims)
    RandomDraw{T, length(dims), typeof(reshaped)}(reshaped, x.nchains)
end

function Base.reshape(x::RandomDraw, dims::Int...)
    reshape(x, dims)
end

function Base.similar(x::RandomDraw{T, N}, ::Type{S}, dims::Dims) where {T, N, S}
    new_dims = (size(x.draws, 1), dims...)
    data = similar(x.draws, S, new_dims)
    RandomDraw{S, length(dims), typeof(data)}(data, x.nchains)
end

function Base.copy(x::RandomDraw{T, N}) where {T, N}
    RandomDraw{T, N, typeof(x.draws)}(copy(x.draws), x.nchains)
end

# `collect`, `Array` and `map` allocate `Array{eltype(x)}` up front and convert each
# element into it. That is the one place the `eltype(x) === T` declaration cannot hold:
# an element of a RandomDraw is a scalar RandomDraw, not a T. Materialise the scalar RVs
# explicitly. (The similar-based fallbacks — x[2:3], vcat, reverse, sum — need no help,
# because `similar` is overridden to return a RandomDraw.)
function Base.collect(x::RandomDraw{T, N}) where {T, N}
    out = [x[i] for i in eachindex(x)]
    return reshape(out, size(x))
end

Base.Array(x::RandomDraw) = collect(x)

# `map(sin, x)` and `sin.(x)` must agree, so route map through broadcast. Unlike
# broadcast, map does not expand singleton dimensions, so check shapes first.
Base.map(f, x::RandomDraw) = broadcast(f, x)

function Base.map(f, x::RandomDraw, ys...)
    for y in ys
        size(y) == size(x) || throw(DimensionMismatch(
            "map requires equal sizes, got $(size(x)) and $(size(y))"))
    end
    return broadcast(f, x, ys...)
end

function Base.show(io::IO, x::RandomDraw{T, N}) where {T, N}
    nd = ndraws(x)
    nc = nchains(x)
    if N == 0
        print(io, "RandomDraw{$T}<$(nd),$(nc)>[1] mean ± sd:")
        if nd > 0
            m = Statistics.mean(x)
            s = Statistics.std(x)
            print(io, "\n[1] ", round(m; digits=2), " ± ", round(s; digits=2))
        end
    else
        sz_str = join(size(x), ",")
        print(io, "RandomDraw{$T}<$(nd),$(nc)>[$(sz_str)] mean ± sd:")
        n_show = min(length(x), 6)
        m_vals = Statistics.mean(x)
        s_vals = Statistics.std(x)
        if N == 1
            for i in 1:n_show
                print(io, "\n[$i] ", round(m_vals[i]; digits=2), " ± ", round(s_vals[i]; digits=2))
            end
        else
            for i in 1:min(size(x, 1), 4)
                print(io, "\n[$i,:] ")
                print(io, round.(m_vals[i, :]; digits=2), " ± ", round.(s_vals[i, :]; digits=2))
            end
            if size(x, 1) > 4
                print(io, "\n...")
            end
        end
        if length(x) > 6
            print(io, "\n...")
        end
    end
end
