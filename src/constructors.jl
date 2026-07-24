"""
    ndraws(x)

Total number of Monte Carlo draws backing `x`, i.e. `niterations(x) * nchains(x)`.
"""
ndraws(x::RandomDraw) = size(x.draws, 1)

"""
    nchains(x)

Number of chains packed into the draws axis of `x`.
"""
nchains(x::RandomDraw) = x.nchains

"""
    niterations(x)

Number of draws per chain, `ndraws(x) ÷ nchains(x)`.
"""
niterations(x::RandomDraw) = ndraws(x) ÷ nchains(x)

"""
    variables(x)

The parameter names carried by `x`, or `nothing` if it has none. Only vector random
variables (`N == 1`) can carry names; see [`from_chains`](@ref).
"""
variables(x::RandomDraw) = x.names

"""
    draws(x; with_chains=false)

The raw draws backing `x`. With `with_chains=false` (default) this is the stored
`(ndraws, shape...)` array. With `with_chains=true` the draws axis is split into
`(niterations, nchains, shape...)`, iterations fastest.
"""
function draws(x::RandomDraw; with_chains::Bool=false)
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

function RandomDraw(x::AbstractVector{T}) where {T}
    RandomDraw{T, 0, typeof(x)}(x, 1)
end

"""
    RandomDraw(x::AbstractArray; nchains=1, with_chains=false, names=nothing)

Wrap an array of draws as a random variable. `x` is `(ndraws, shape...)` and the result
has shape `size(x)[2:end]`.

Pass `nchains` to declare how many chains are packed into the draws axis (it must divide
`ndraws`). Pass `with_chains=true` when `x` is instead `(iterations, chains, shape...)`;
the first two axes are then flattened into the draws axis and `nchains` is taken from the
data (an explicit `nchains` is ignored, with a warning).

Pass `names` (a vector of `Symbol`s) to label the elements of a vector random variable;
this is only valid when the result has `N == 1`.
"""
function RandomDraw(x::AbstractArray{T}; nchains::Int=1, with_chains::Bool=false,
                    names::Union{Nothing, AbstractVector{Symbol}}=nothing) where {T}
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
        return RandomDraw{T, n_out, typeof(reshaped)}(reshaped, n_chain, names)
    end
    if ndims(x) == 1
        # A flat vector is a scalar RV whose draws are the whole vector; honor nchains.
        return RandomDraw{T, 0, typeof(x)}(x, nchains, names)
    end
    sz = size(x)
    n_out = length(sz) - 1
    return RandomDraw{T, n_out, typeof(x)}(x, nchains, names)
end

function RandomDraw(x::RandomDraw)
    return x
end

"""
    as_rs(x)

Lift a constant number or array `x` to a `RandomDraw` with a single draw, preserving its
element type. Useful for combining constants with random variables in arithmetic; a
single-draw operand is treated as chain-agnostic and broadcasts against any number of
draws.
"""
function as_rs(x::AbstractArray{T}) where {T}
    sz = size(x)
    A = reshape(x, 1, sz...)
    RandomDraw{T, length(sz), typeof(A)}(A, 1)
end

function as_rs(x::Number)
    data = fill(x, 1)
    RandomDraw{typeof(x), 0, typeof(data)}(data, 1)
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
    RandomDraw{eltype(reshaped), 1, typeof(reshaped)}(reshaped, 1)
end

function Base.rand(rng::AbstractRNG, ::Type{RandomDraw{T, 0}}, n_draws::Int=2000) where {T}
    data = rand(rng, T, n_draws)
    RandomDraw(data)
end

function Base.rand(rng::AbstractRNG, ::Type{RandomDraw{T, N}}, dims::Dims, n_draws::Int=2000) where {T, N}
    length(dims) == N || error("Requested RandomDraw{$T, $N} but dims=$dims has $(length(dims)) dimensions")
    sz = (n_draws, dims...)
    data = rand(rng, T, sz)
    RandomDraw{T, N, typeof(data)}(data, 1)
end
