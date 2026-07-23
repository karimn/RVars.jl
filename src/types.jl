"""
    RandomDraw{T, N, A} <: AbstractArray{T, N}

A random variable represented by Monte Carlo draws (e.g. a posterior sample), in the
spirit of R's `posterior::rvar`.

A `RandomDraw{T, N}` behaves as an `N`-dimensional array of scalar random variables with
scalar element type `T`. The draws are stored in the wrapped array `A`, which has **one
more dimension** than the value presents:

- axis 1 of `A` is the draws axis (the Monte Carlo samples), always present;
- axes `2:N+1` are the visible shape returned by [`size`](@ref).

So `N = 0` wraps a vector (a scalar RV), `N = 1` wraps a matrix (a vector RV), and so on.
Indexing an element (`x[i]`, `x[i, j]`) returns a scalar `RandomDraw{T, 0}` holding all
draws of that element.

Multiple chains are packed into the draws axis with iterations fastest and chains slowest,
so `ndraws == niterations * nchains`. Use [`draws(x; with_chains=true)`](@ref draws) to
recover the `(iterations, chains, shape...)` view.

Construct with [`RandomDraw`](@ref RandomDraw(::AbstractArray)), [`as_rs`](@ref),
[`rvar_rng`](@ref), or [`from_chains`](@ref); reduce over draws with `mean`/`std`/`quantile`
or [`E`](@ref)/[`Pr`](@ref); reduce over the element shape (per draw) with the `rs_*`
functions.

A vector random variable (`N == 1`) may additionally carry parameter names; see
[`variables`](@ref) and [`from_chains`](@ref). Names are dropped by any operation that
does not preserve elementwise identity.

# Deviations from the `AbstractArray` contract

`RandomDraw` subtypes `AbstractArray` to reuse Julia's indexing and `similar` plumbing,
not to promise the full contract. Two deliberate deviations:

- `eltype(x) === T`, but `x[i]` returns a `RandomDraw{T, 0}` — every element is itself a
  random variable. `collect`, `Array` and `map` have explicit methods because of this;
  everything routed through `similar` works unchanged. Use [`draws`](@ref) for the raw
  `(ndraws, shape...)` store.
- `==`, `<`, `<=`, `>`, `>=` compare draw-by-draw and return a `RandomDraw{Bool}`, not a
  `Bool` (matching R's `rvar`). Use `isequal` for a `Bool`-returning identity test, which
  is what `Dict` and `in` need.
"""
struct RandomDraw{T, N, A <: AbstractArray{T}} <: AbstractArray{T, N}
    draws::A
    nchains::Int
    names::Union{Nothing, Vector{Symbol}}

    function RandomDraw{T, N, A}(draws::A, nchains::Int=1,
                                 names::Union{Nothing, AbstractVector{Symbol}}=nothing
                                 ) where {T, N, A <: AbstractArray{T}}
        nchains >= 1 || error("nchains must be >= 1")
        nd = size(draws, 1)
        if nd % nchains != 0
            error("Number of chains ($nchains) does not divide number of draws ($nd)")
        end
        expected_dims = N + 1
        actual_dims = ndims(draws)
        if actual_dims != expected_dims
            error("Expected draws array with $expected_dims dimensions (draws × shape), got $actual_dims")
        end
        if names !== nothing
            # Names label the elements of a single logical axis. N == 0 has no elements to
            # label; N >= 2 would need one name vector per axis, a different feature.
            N == 1 || error("Parameter names are only supported for vector random variables (N == 1), got N == $N")
            length(names) == size(draws, 2) ||
                error("Got $(length(names)) names for a length-$(size(draws, 2)) random variable")
        end
        new{T, N, A}(draws, nchains, names === nothing ? nothing : collect(Symbol, names))
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
