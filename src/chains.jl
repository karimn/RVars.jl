"""
    from_chains(array)
    from_chains(array, param_names; flat=false)

Build random variables from a 3-D `(iterations, variables, chains)` array — the layout
produced by MCMC samplers. Iterations and chains are folded into the draws axis (iterations
fastest) so per-variable draws stay contiguous, and `nchains` is taken from the data.

Without `param_names` the result is a single `RVar{T, 1}` of length `variables`: one
element per column of the array, unnamed.

With `param_names` (a vector of `Symbol`s or `String`s, one entry per variable) the result
is instead a `NamedTuple` of one random variable per *model parameter*, with array-valued
parameters reassembled into their declared shape — see [`rvars`](@ref), which does the
regrouping. Pass `flat=true` to get the ungrouped `RVar{T, 1}` carrying `param_names`
verbatim (readable with [`variables`](@ref)), one element per column.

With the `MCMCChains` extension loaded, `from_chains(::MCMCChains.Chains)` (and the
`RVar(::Chains)` constructor) accept a `Chains` object directly, taking the parameter
names off the object.
"""
function from_chains(array::AbstractArray{T, 3}) where {T}
    n_iter, n_var, n_chain = size(array, 1), size(array, 2), size(array, 3)
    # Input is (iteration, variable, chain). Move chain next to iteration so the
    # two draw axes are contiguous and column-major reshape gathers, per variable,
    # all iterations of chain 1, then chain 2, ... (matching draws(; with_chains)).
    permuted = permutedims(array, (1, 3, 2))
    reshaped = reshape(permuted, n_iter * n_chain, n_var)
    RVar{T, 1, typeof(reshaped)}(reshaped, n_chain)
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{Symbol};
                     flat::Bool=false) where {T}
    rd = from_chains(array)
    d = draws(rd)
    named = RVar{T, 1, typeof(d)}(d, nchains(rd), param_names)
    return flat ? named : rvars(named)
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{String};
                     flat::Bool=false) where {T}
    from_chains(array, Symbol.(param_names); flat=flat)
end
