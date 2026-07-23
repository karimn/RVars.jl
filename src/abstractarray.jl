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

# Subsetting a vector RV. The generic AbstractArray fallback would route through
# `similar`, which receives only a type and a shape and so cannot carry the names;
# slicing the flat store directly both preserves them and skips the fallback's
# element-by-element loop.
function Base.getindex(x::RandomDraw{T, 1}, I::AbstractVector{<:Integer}) where {T}
    @boundscheck checkbounds(x, I)
    data = _flat_store(x)[:, I]
    nms = x.names === nothing ? nothing : x.names[I]
    RandomDraw{T, 1, typeof(data)}(data, x.nchains, nms)
end

Base.getindex(x::RandomDraw{T, 1}, ::Colon) where {T} = x[1:length(x)]

# Bool <: Integer, so without this method a logical index would silently dispatch to the
# integer method above and be read as positions 1 and 0. Julia reports no ambiguity here.
function Base.getindex(x::RandomDraw{T, 1}, idx::AbstractVector{Bool}) where {T}
    length(idx) == length(x) ||
        throw(DimensionMismatch("logical index length $(length(idx)) != length $(length(x))"))
    return x[findall(idx)]
end

function _name_index(x::RandomDraw, s::Symbol)
    nms = x.names
    nms === nothing && error("This RandomDraw has no parameter names")
    i = findfirst(isequal(s), nms)
    i === nothing &&
        error("Unknown parameter name :$s; available: $(join(string.(nms), ", "))")
    return i
end

Base.getindex(x::RandomDraw{T, 1}, s::Symbol) where {T} = x[_name_index(x, s)]

function Base.getindex(x::RandomDraw{T, 1}, S::AbstractVector{Symbol}) where {T}
    return x[[_name_index(x, s) for s in S]]
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
    RandomDraw{T, N, typeof(x.draws)}(copy(x.draws), x.nchains, x.names)
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

# `==` is elementwise by design (src/arithmetic.jl), so it cannot answer "are these the
# same random variable?". `isequal` does, and is what Dict and `in` dispatch on.
function Base.isequal(x::RandomDraw, y::RandomDraw)
    return isequal(x.draws, y.draws) && x.nchains == y.nchains && isequal(x.names, y.names)
end

# Base's hash(::AbstractArray) hashes elements, and an element of a RandomDraw is
# another RandomDraw — for N == 0 that is itself, so the generic method recurses until
# the stack overflows. Hash the fields directly, matching what isequal compares.
Base.hash(x::RandomDraw, h::UInt) = hash(x.names, hash(x.nchains, hash(x.draws, hash(:RandomDraw, h))))

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
            nms = x.names
            for i in 1:n_show
                lbl = nms === nothing ? string(i) : string(nms[i])
                print(io, "\n[", lbl, "] ",
                      round(m_vals[i]; digits=2), " ± ", round(s_vals[i]; digits=2))
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
