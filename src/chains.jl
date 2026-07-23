function from_chains(array::AbstractArray{T, 3}) where {T}
    n_iter, n_var, n_chain = size(array, 1), size(array, 2), size(array, 3)
    # Input is (iteration, variable, chain). Move chain next to iteration so the
    # two draw axes are contiguous and column-major reshape gathers, per variable,
    # all iterations of chain 1, then chain 2, ... (matching draws(; with_chains)).
    permuted = permutedims(array, (1, 3, 2))
    reshaped = reshape(permuted, n_iter * n_chain, n_var)
    RandomDraw{T, 1, typeof(reshaped)}(reshaped, n_chain)
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{Symbol}) where {T}
    rd = from_chains(array)
    rd, param_names
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{String}) where {T}
    from_chains(array, Symbol.(param_names))
end
