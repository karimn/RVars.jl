function _summarise_by_element(x::RVar, f::Function)
    d = draws(x)
    n_draws = size(d, 1)
    nd = ndims(d)
    if nd == 1
        return f(d)
    end
    rest_shape = size(d)[2:end]
    n_elems = prod(rest_shape)
    flat = reshape(d, n_draws, n_elems)
    # Applying f down each element's draws yields an (L, n_elems) array, where L is the
    # length of f's output (1 for a scalar reduction, length(probs) for a vector one).
    result = mapslices(f, flat; dims=1)
    L = size(result, 1)
    if L == 1
        return reshape(dropdims(result; dims=1), rest_shape...)
    end
    return reshape(result, L, rest_shape...)
end

Statistics.mean(x::RVar) = _summarise_by_element(x, Statistics.mean)

function Statistics.std(x::RVar; corrected::Bool=true)
    _summarise_by_element(x, v -> Statistics.std(v; corrected=corrected))
end

function Statistics.var(x::RVar; corrected::Bool=true)
    _summarise_by_element(x, v -> Statistics.var(v; corrected=corrected))
end

Statistics.median(x::RVar) = _summarise_by_element(x, Statistics.median)
Base.minimum(x::RVar) = _summarise_by_element(x, Base.minimum)
Base.maximum(x::RVar) = _summarise_by_element(x, Base.maximum)

function Statistics.quantile(x::RVar, p::Union{AbstractVector, Real})
    _summarise_by_element(x, v -> Statistics.quantile(v, p))
end

"""
    E(x)

Expectation of `x`, estimated as the mean over draws. Alias for `mean(x)`: returns a plain
scalar/array of per-element means (one value per element of the visible shape), not a
`RVar`.
"""
E(x::RVar) = Statistics.mean(x)

"""
    Pr(x)

Probability that the boolean random variable `x` holds, estimated as the fraction of draws
that are `true` (per element). Requires a `RVar` with `Bool` draws, e.g. the result
of a comparison like `x .> 0`.
"""
function Pr(x::RVar{<:Any, <:Any, <:AbstractArray{Bool}})
    Statistics.mean(x)
end

"""
    rs_mean(x), rs_sum(x), rs_sd(x), rs_var(x), rs_median(x), rs_min(x), rs_max(x)
    rs_quantile(x, probs)

Reduce-shape reductions: collapse the element (visible) shape of `x` *per draw*, keeping the
draws axis, and return a `RVar`. For example `rs_mean(x)` is the across-element mean
for each draw — a scalar RV that still carries the full posterior. `rs_quantile` returns a
length-`length(probs)` vector RV of per-draw quantiles.

Contrast with `mean`/`std`/`quantile`/[`E`](@ref)/[`Pr`](@ref), which reduce over draws
per element and return plain numbers/arrays.
"""
function rs_mean(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.mean(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_sum(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = sum(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_sd(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.std(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_var(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.var(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_median(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.median(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_min(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Base.minimum(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_max(x::RVar{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Base.maximum(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RVar(reshape(result, n_draws), nchains=nchains(x))
end

function rs_quantile(x::RVar{T, N}, probs::AbstractVector) where {T, N}
    # Like the other rs_ reductions: collapse the element shape per draw (here into the
    # requested quantiles), keeping the draws axis. Result is a length-(probs) vector RV.
    d = draws(x)
    n_draws = size(d, 1)
    n_elems = prod(size(d)[2:end])
    flat = reshape(d, n_draws, n_elems)
    L = length(probs)
    result = similar(flat, float(eltype(flat)), (n_draws, L))
    for i in 1:n_draws
        result[i, :] .= Statistics.quantile(view(flat, i, :), probs)
    end
    RVar{eltype(result), 1, typeof(result)}(result, nchains(x))
end
