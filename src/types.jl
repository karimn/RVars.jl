"""
    RVar{T, N, A} <: AbstractArray{T, N}

A random variable represented by Monte Carlo draws (e.g. a posterior sample), in the
spirit of R's `posterior::rvar`.

A `RVar{T, N}` behaves as an `N`-dimensional array of scalar random variables with
scalar element type `T`. The draws are stored in the wrapped array `A`, which has **one
more dimension** than the value presents:

- axis 1 of `A` is the draws axis (the Monte Carlo samples), always present;
- axes `2:N+1` are the visible shape returned by [`size`](@ref).

So `N = 0` wraps a vector (a scalar RV), `N = 1` wraps a matrix (a vector RV), and so on.
Indexing an element (`x[i]`, `x[i, j]`) returns a scalar `RVar{T, 0}` holding all
draws of that element.

Multiple chains are packed into the draws axis with iterations fastest and chains slowest,
so `ndraws == niterations * nchains`. Use [`draws(x; with_chains=true)`](@ref draws) to
recover the `(iterations, chains, shape...)` view.

Construct with [`RVar`](@ref RVar(::AbstractArray)), [`as_rs`](@ref),
[`rvar_rng`](@ref), or [`from_chains`](@ref); reduce over draws with `mean`/`std`/`quantile`
or [`E`](@ref)/[`Pr`](@ref); reduce over the element shape (per draw) with the `rs_*`
functions.

A vector random variable (`N == 1`) may additionally carry parameter names; see
[`variables`](@ref) and [`from_chains`](@ref). Names are dropped by any operation that
does not preserve elementwise identity.

A random variable of any rank may carry **dimension metadata**: one name per axis
([`dimnames`](@ref)) and, optionally, a vector of labels along each axis
([`dimlabels`](@ref)). These describe the *axes*, where `names` describes the *elements*,
so a parameter declared `a[trial, arm]` can report `(:trial, :arm)` with `:arm` running
over `["control", "drug"]` and be indexed as `a[trial=1, arm=:drug]`. A chain records only
`a[1,2]`, so this is supplied at extraction — see [`rvars`](@ref). Dimension metadata
survives slicing (a scalar-indexed axis drops out, and a sliced axis's labels are subset
along with it) and shape-preserving arithmetic; it is dropped by anything that changes the
rank, and by combining two random variables that describe their axes differently.

# Deviations from the `AbstractArray` contract

`RVar` subtypes `AbstractArray` to reuse Julia's indexing and `similar` plumbing,
not to promise the full contract. Two deliberate deviations:

- `eltype(x) === T`, but `x[i]` returns a `RVar{T, 0}` — every element is itself a
  random variable. `collect`, `Array` and `map` have explicit methods because of this;
  everything routed through `similar` works unchanged. Use [`draws`](@ref) for the raw
  `(ndraws, shape...)` store. The *typed* materialization forms are not among these —
  `collect(Float64, x)`, `convert(Array{Float64}, x)` and `Array{Float64}(x)` still throw
  `MethodError`.
- `==`, `<`, `<=`, `>`, `>=` compare draw-by-draw and return a `RVar{Bool}`, not a
  `Bool` (matching R's `rvar`). Use `isequal` for a `Bool`-returning identity test, which
  is what `Dict` and `Set` need.
"""
struct RVar{T, N, A <: AbstractArray{T}} <: AbstractArray{T, N}
    draws::A
    nchains::Int
    names::Union{Nothing, Vector{Symbol}}
    dimnames::Union{Nothing, NTuple{N, Symbol}}
    dimlabels::Union{Nothing, NTuple{N, Union{Nothing, Vector}}}

    function RVar{T, N, A}(draws::A, nchains::Int=1,
                                 names::Union{Nothing, AbstractVector{Symbol}}=nothing,
                                 dimnames::Union{Nothing, Tuple{Vararg{Symbol}}}=nothing,
                                 dimlabels::Union{Nothing, Tuple{Vararg{Union{Nothing, AbstractVector}}}}=nothing
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
        if dimnames !== nothing
            # One name per *axis*, unlike `names` which is one per element.
            length(dimnames) == N ||
                error("Got $(length(dimnames)) dimension names for a $N-dimensional random variable")
            allunique(dimnames) ||
                error("Dimension names must be unique, got $(dimnames)")
        end
        if dimlabels !== nothing
            # One optional label vector per axis, each as long as that axis. Labels are what
            # let :arm mean "control"/"drug" rather than 1/2, so a length mismatch is an
            # error here rather than a mislabeled axis discovered at plotting time.
            length(dimlabels) == N ||
                error("Got $(length(dimlabels)) label vectors for a $N-dimensional random variable")
            for (d, labs) in enumerate(dimlabels)
                labs === nothing && continue
                len = size(draws, d + 1)
                length(labs) == len || error(
                    "Got $(length(labs)) labels for axis $d, which has length $len")
                allunique(labs) || error("Labels for axis $d must be unique, got $(labs)")
            end
        end
        new{T, N, A}(draws, nchains, names === nothing ? nothing : collect(Symbol, names),
                     dimnames === nothing ? nothing : NTuple{N, Symbol}(dimnames),
                     dimlabels === nothing ? nothing :
                         NTuple{N, Union{Nothing, Vector}}(map(l -> l === nothing ? nothing : collect(l),
                                                               dimlabels)))
    end
end

Base.eltype(::Type{<:RVar{T}}) where {T} = T
Base.ndims(::Type{<:RVar{T, N}}) where {T, N} = N

function Base.size(x::RVar)
    n_extra = ndims(x.draws) - 1
    ntuple(i -> size(x.draws, i + 1), n_extra)
end

function Base.size(x::RVar, d::Int)
    nd = ndims(x.draws) - 1
    if d <= nd
        return size(x.draws, d + 1)
    end
    return 1
end

Base.length(x::RVar) = prod(size(x))

function Base.axes(x::RVar)
    n_extra = ndims(x.draws) - 1
    ntuple(i -> axes(x.draws, i + 1), n_extra)
end

function Base.axes(x::RVar, d::Int)
    nd = ndims(x.draws) - 1
    if d <= nd
        return axes(x.draws, d + 1)
    end
    return Base.OneTo(1)
end

function _broadcast_draws(x::RVar, n::Int)
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
        x isa RVar || continue
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

# Combine parameter names across operands of an elementwise/broadcast operation, with the
# same rule as _combine_nchains: a single-draw operand is a constant whose names are
# weaker evidence, used only when no multi-draw operand supplies any; two genuine but
# differing name sets cannot be reconciled, so the result is unnamed.
function _combine_names(operands...)
    nms = nothing
    seen = false
    const_nms = nothing
    const_seen = false
    for x in operands
        x isa RVar || continue
        if size(x.draws, 1) == 1
            if !const_seen
                const_nms = x.names
                const_seen = true
            elseif !isequal(const_nms, x.names)
                const_nms = nothing
            end
            continue
        end
        if !seen
            nms = x.names
            seen = true
        elseif !isequal(nms, x.names)
            nms = nothing
        end
    end
    return seen ? nms : const_nms
end

# Combine dimension metadata (axis names and axis labels) across operands of an
# elementwise/broadcast operation. Unlike `names`, an operand carrying no metadata is not
# evidence of disagreement — it simply has nothing to say, so `a .+ sigma` keeps a's axes.
# Only two operands that both describe their axes, differently, force the result to drop
# them. Returns a (dimnames, dimlabels) pair.
function _combine_dimmeta(operands...)
    dn = nothing
    dl = nothing
    dn_conflict = false
    dl_conflict = false
    for x in operands
        x isa RVar || continue
        if x.dimnames !== nothing
            if dn === nothing
                dn = x.dimnames
            elseif !isequal(dn, x.dimnames)
                dn_conflict = true
            end
        end
        if x.dimlabels !== nothing
            if dl === nothing
                dl = x.dimlabels
            elseif !isequal(dl, x.dimlabels)
                dl_conflict = true
            end
        end
    end
    return (dn_conflict ? nothing : dn, dl_conflict ? nothing : dl)
end

# Attach `nms` to `x` only where it still describes the result elementwise. Broadcasting
# can change both rank and length (a named length-3 vector RV times a 3x2 matrix RV gives
# an N=2 result), and stale names are worse than none. Dimension names and labels are held
# to the same standard: each is kept only while it still matches the result's rank, and
# labels additionally only while every labelled axis still has its original length.
function _maybe_names(x::RVar{T, N}, nms, dn=nothing, dl=nothing) where {T, N}
    keep_nms = !(nms === nothing || N != 1 || length(nms) != length(x))
    keep_dn = dn !== nothing && length(dn) == N
    keep_dl = dl !== nothing && length(dl) == N &&
              all(d -> dl[d] === nothing || length(dl[d]) == size(x, d), 1:N)
    (keep_nms || keep_dn || keep_dl) || return x
    return RVar{T, N, typeof(x.draws)}(x.draws, x.nchains,
                                       keep_nms ? nms : nothing,
                                       keep_dn ? dn : nothing,
                                       keep_dl ? dl : nothing)
end
