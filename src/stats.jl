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
    result = mapslices(f, flat; dims=1)
    sz = size(result)
    if length(sz) > 1 && sz[1] == 1
        result = dropdims(result; dims=1)
    end
    if isempty(rest_shape)
        return result
    end
    return reshape(result, rest_shape...)
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
    d = draws(x)
    n_draws = size(d, 1)
    rest = size(d)[2:end]
    n_elems = prod(rest)
    flat = reshape(d, n_draws, n_elems)
    qs = [Statistics.quantile(flat[:, j], probs) for j in 1:n_elems]
    q_matrix = hcat(qs...)
    replicated = repeat(q_matrix; inner=(n_draws, 1))
    RandomDraw(reshape(replicated, n_draws, length(probs), rest...), nchains=nchains(x))
end
