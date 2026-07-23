function _summarise_by_element(x::RandomDraw, f::Function)
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

Statistics.mean(x::RandomDraw) = _summarise_by_element(x, Statistics.mean)

function Statistics.std(x::RandomDraw; corrected::Bool=true)
    _summarise_by_element(x, v -> Statistics.std(v; corrected=corrected))
end

function Statistics.var(x::RandomDraw; corrected::Bool=true)
    _summarise_by_element(x, v -> Statistics.var(v; corrected=corrected))
end

Statistics.median(x::RandomDraw) = _summarise_by_element(x, Statistics.median)
Base.minimum(x::RandomDraw) = _summarise_by_element(x, Base.minimum)
Base.maximum(x::RandomDraw) = _summarise_by_element(x, Base.maximum)

function Statistics.quantile(x::RandomDraw, p::Union{AbstractVector, Real})
    _summarise_by_element(x, v -> Statistics.quantile(v, p))
end

E(x::RandomDraw) = Statistics.mean(x)

function Pr(x::RandomDraw{<:Any, <:Any, <:AbstractArray{Bool}})
    Statistics.mean(x)
end

function rs_mean(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.mean(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_sum(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = sum(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_sd(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.std(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_var(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.var(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_median(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Statistics.median(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_min(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Base.minimum(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_max(x::RandomDraw{T, N}) where {T, N}
    d = draws(x)
    n_draws = size(d, 1)
    result = Base.maximum(d; dims=ntuple(i -> i + 1, N))
    result = dropdims(result; dims=ntuple(i -> i + 1, N))
    RandomDraw(reshape(result, n_draws), nchains=nchains(x))
end

function rs_quantile(x::RandomDraw{T, N}, probs::AbstractVector) where {T, N}
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
    RandomDraw{eltype(result), 1, typeof(result)}(result, nchains(x))
end
