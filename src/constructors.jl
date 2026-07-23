ndraws(x::RandomDraw) = size(x.draws, 1)
nchains(x::RandomDraw) = x.nchains
niterations(x::RandomDraw) = ndraws(x) ÷ nchains(x)

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

function RandomDraw(x::AbstractArray{T}; nchains::Int=1, with_chains::Bool=false) where {T}
    if with_chains
        sz = size(x)
        length(sz) >= 2 || error("with_chains=true requires >= 2 dims (iterations, chains, ...)")
        n_iter, n_chain = sz[1], sz[2]
        rest = sz[3:end]
        new_sz = (n_iter * n_chain, rest...)
        reshaped = reshape(x, new_sz)
        n_out = length(rest)
        return RandomDraw{T, n_out, typeof(reshaped)}(reshaped, n_chain)
    end
    if ndims(x) == 1
        # A flat vector is a scalar RV whose draws are the whole vector; honor nchains.
        return RandomDraw{T, 0, typeof(x)}(x, nchains)
    end
    sz = size(x)
    n_out = length(sz) - 1
    return RandomDraw{T, n_out, typeof(x)}(x, nchains)
end

function RandomDraw(x::RandomDraw)
    return x
end

function as_rs(x::AbstractArray{T}) where {T}
    sz = size(x)
    A = reshape(x, 1, sz...)
    RandomDraw{T, length(sz), typeof(A)}(A, 1)
end

function as_rs(x::Number)
    data = fill(x, 1)
    RandomDraw{typeof(x), 0, typeof(data)}(data, 1)
end

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
