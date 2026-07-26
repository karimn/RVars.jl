"""
    ndraws(x)

Total number of Monte Carlo draws backing `x`, i.e. `niterations(x) * nchains(x)`.
"""
ndraws(x::RVar) = size(x.draws, 1)

"""
    nchains(x)

Number of chains packed into the draws axis of `x`.
"""
nchains(x::RVar) = x.nchains

"""
    niterations(x)

Number of draws per chain, `ndraws(x) ÷ nchains(x)`.
"""
niterations(x::RVar) = ndraws(x) ÷ nchains(x)

"""
    variables(x)

The parameter names carried by `x`, or `nothing` if it has none. Only vector random
variables (`N == 1`) can carry names; see [`from_chains`](@ref).
"""
variables(x::RVar) = x.names

"""
    dimnames(x)

The names of `x`'s dimensions as an `N`-tuple of `Symbol`s, or `nothing` if it has none.

Unlike [`variables`](@ref), which names one *element* of a vector random variable, these
name the *axes*: a parameter declared `a[trial, patient]` can carry `(:trial, :patient)`.
Supply them when extracting from a fit — see [`rvars`](@ref) — since a chain records only
`a[1,2]` and not what those positions mean.
"""
dimnames(x::RVar) = x.dimnames

"""
    dimlabels(x)
    dimlabels(x, dim)

The labels along each of `x`'s dimensions, as an `N`-tuple whose entries are either a
vector of labels or `nothing` for an unlabelled axis. Returns `nothing` if `x` carries no
labels at all.

Labels are what let an axis be `["control", "drug"]` rather than `1:2`, so
`x[arm=:drug]` and printed output can speak in the model's own terms. The second form
takes a dimension name (or position) and returns just that axis's labels.
"""
dimlabels(x::RVar) = x.dimlabels

function dimlabels(x::RVar, dim::Symbol)
    d = _dim_index(x, dim)
    return x.dimlabels === nothing ? nothing : x.dimlabels[d]
end

function dimlabels(x::RVar, dim::Integer)
    1 <= dim <= ndims(x) || throw(BoundsError(x, dim))
    return x.dimlabels === nothing ? nothing : x.dimlabels[dim]
end

# Resolve a dimension name to its axis position.
function _dim_index(x::RVar, dim::Symbol)
    dn = x.dimnames
    dn === nothing && error("This RVar has no dimension names; supply them with rvars(...; dims=...)")
    d = findfirst(isequal(dim), dn)
    if d === nothing
        avail = join(string.(dn), ", ")
        error("Unknown dimension :$dim; available: $avail")
    end
    return d
end

"""
    draws(x; with_chains=false)

The raw draws backing `x`. With `with_chains=false` (default) this is the stored
`(ndraws, shape...)` array. With `with_chains=true` the draws axis is split into
`(niterations, nchains, shape...)`, iterations fastest.
"""
function draws(x::RVar; with_chains::Bool=false)
    if with_chains
        d = x.draws
        sz = size(d)
        nit = niterations(x)
        nc = nchains(x)
        rest = sz[2:end]
        new_sz = (nit, nc, rest...)
        return reshape(d, new_sz)
    else
        return x.draws
    end
end

function RVar(x::AbstractVector{T}) where {T}
    RVar{T, 0, typeof(x)}(x, 1)
end

"""
    RVar(x::AbstractArray; nchains=1, with_chains=false, names=nothing)

Wrap an array of draws as a random variable. `x` is `(ndraws, shape...)` and the result
has shape `size(x)[2:end]`.

Pass `nchains` to declare how many chains are packed into the draws axis (it must divide
`ndraws`). Pass `with_chains=true` when `x` is instead `(iterations, chains, shape...)`;
the first two axes are then flattened into the draws axis and `nchains` is taken from the
data (an explicit `nchains` is ignored, with a warning).

Pass `names` (a vector of `Symbol`s) to label the elements of a vector random variable;
this is only valid when the result has `N == 1`.

Pass `dimnames` (a tuple of `Symbol`s, one per visible axis) to name the dimensions, and
`dimlabels` (a tuple with one entry per axis, each either a vector of labels as long as
that axis or `nothing`) to label the positions along them. See [`dimnames`](@ref) and
[`dimlabels`](@ref).
"""
function RVar(x::AbstractArray{T}; nchains::Int=1, with_chains::Bool=false,
                    names::Union{Nothing, AbstractVector{Symbol}}=nothing,
                    dimnames::Union{Nothing, Tuple{Vararg{Symbol}}}=nothing,
                    dimlabels::Union{Nothing, Tuple{Vararg{Union{Nothing, AbstractVector}}}}=nothing) where {T}
    if with_chains
        if nchains != 1
            @warn "with_chains=true derives nchains from the data; ignoring nchains=$nchains"
        end
        sz = size(x)
        length(sz) >= 2 || error("with_chains=true requires >= 2 dims (iterations, chains, ...)")
        n_iter, n_chain = sz[1], sz[2]
        rest = sz[3:end]
        new_sz = (n_iter * n_chain, rest...)
        reshaped = reshape(x, new_sz)
        n_out = length(rest)
        return RVar{T, n_out, typeof(reshaped)}(reshaped, n_chain, names, dimnames, dimlabels)
    end
    if ndims(x) == 1
        # A flat vector is a scalar RV whose draws are the whole vector; honor nchains.
        return RVar{T, 0, typeof(x)}(x, nchains, names, dimnames, dimlabels)
    end
    sz = size(x)
    n_out = length(sz) - 1
    return RVar{T, n_out, typeof(x)}(x, nchains, names, dimnames, dimlabels)
end

function RVar(x::RVar)
    return x
end

"""
    as_rs(x)

Lift a constant number or array `x` to a `RVar` with a single draw, preserving its
element type. Useful for combining constants with random variables in arithmetic; a
single-draw operand is treated as chain-agnostic and broadcasts against any number of
draws.
"""
function as_rs(x::AbstractArray{T}) where {T}
    sz = size(x)
    A = reshape(x, 1, sz...)
    RVar{T, length(sz), typeof(A)}(A, 1)
end

function as_rs(x::Number)
    data = fill(x, 1)
    RVar{typeof(x), 0, typeof(data)}(data, 1)
end

"""
    rvar_rng(rng_func, n, args...; ndraws=2000, kwargs...)

Build a length-`n` vector random variable by drawing samples from `rng_func`. `rng_func`
is called as `rng_func(ndraws * n, args...; kwargs...)` and must return a flat collection of
that length (e.g. `randn`, or `k -> rand(1:6, k)`); the result's element type is preserved.
"""
function rvar_rng(rng_func::Function, n::Int, args...; ndraws::Int=2000, kwargs...)
    result = rng_func(ndraws * n, args...; kwargs...)
    reshaped = reshape(result, ndraws, n)
    RVar{eltype(reshaped), 1, typeof(reshaped)}(reshaped, 1)
end

function Base.rand(rng::AbstractRNG, ::Type{RVar{T, 0}}, n_draws::Int=2000) where {T}
    data = rand(rng, T, n_draws)
    RVar(data)
end

function Base.rand(rng::AbstractRNG, ::Type{RVar{T, N}}, dims::Dims, n_draws::Int=2000) where {T, N}
    length(dims) == N || error("Requested RVar{$T, $N} but dims=$dims has $(length(dims)) dimensions")
    sz = (n_draws, dims...)
    data = rand(rng, T, sz)
    RVar{T, N, typeof(data)}(data, 1)
end
