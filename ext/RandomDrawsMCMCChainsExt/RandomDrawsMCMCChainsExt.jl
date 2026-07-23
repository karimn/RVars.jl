module RandomDrawsMCMCChainsExt

using RandomDraws
using MCMCChains

function RandomDraw(chn::MCMCChains.Chains)
    arr = MCMCChains.value(chn)
    # arr is (iterations, chains, params)
    n_iter, n_chain, n_var = size(arr, 1), size(arr, 2), size(arr, 3)
    reshaped = reshape(arr, n_iter * n_chain, n_var)
    T = eltype(reshaped)
    RandomDraw{T, 1, typeof(reshaped)}(reshaped, n_chain)
end

function from_chains(chn::MCMCChains.Chains)
    RandomDraw(chn)
end

end
