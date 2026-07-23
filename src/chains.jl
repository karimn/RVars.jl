function from_chains(array::AbstractArray{T, 3}) where {T}
    n_iter, n_var, n_chain = size(array, 1), size(array, 2), size(array, 3)
    reshaped = reshape(array, n_iter * n_chain, n_var)
    RandomDraw{T, 1, typeof(reshaped)}(reshaped, n_chain)
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{Symbol}) where {T}
    rd = from_chains(array)
    rd, param_names
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{String}) where {T}
    from_chains(array, Symbol.(param_names))
end
