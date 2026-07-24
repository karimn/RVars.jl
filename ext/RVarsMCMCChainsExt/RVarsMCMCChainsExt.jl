module RVarsMCMCChainsExt

import RVars
import MCMCChains
import MCMCChains: Chains

# NOTE: methods must be defined with fully-qualified names (RVars.RVar,
# RVars.from_chains) so they EXTEND the parent package's functions. A bare
# `using RVars` + `function RVar(...)` defines shadow functions local to
# this extension module and never dispatches from user code.

function RVars.RVar(chn::Chains)
    # chn.value is an AxisArray indexed (iteration, variable, chain).
    arr = Array(chn.value)
    # MCMCChains.names returns the parameter names in the same order as the :var axis,
    # so they line up with the variable axis of `arr` with no extra bookkeeping.
    # Delegate to the plain-array path, which handles the (iter, var, chain)
    # permutedims + reshape and the nchains bookkeeping.
    RVars.from_chains(arr, MCMCChains.names(chn))
end

RVars.from_chains(chn::Chains) = RVars.RVar(chn)

end
