module RandomDrawsMCMCChainsExt

import RandomDraws
import MCMCChains
import MCMCChains: Chains

# NOTE: methods must be defined with fully-qualified names (RandomDraws.RandomDraw,
# RandomDraws.from_chains) so they EXTEND the parent package's functions. A bare
# `using RandomDraws` + `function RandomDraw(...)` defines shadow functions local to
# this extension module and never dispatches from user code.

function RandomDraws.RandomDraw(chn::Chains)
    # chn.value is an AxisArray indexed (iteration, variable, chain).
    arr = Array(chn.value)
    # MCMCChains.names returns the parameter names in the same order as the :var axis,
    # so they line up with the variable axis of `arr` with no extra bookkeeping.
    # Delegate to the plain-array path, which handles the (iter, var, chain)
    # permutedims + reshape and the nchains bookkeeping.
    RandomDraws.from_chains(arr, MCMCChains.names(chn))
end

RandomDraws.from_chains(chn::Chains) = RandomDraws.RandomDraw(chn)

end
