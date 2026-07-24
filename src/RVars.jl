module RVars

using LinearAlgebra
using Statistics
using Random

export RVar, as_rs, draws, ndraws, nchains, niterations, variables
export rs_mean, rs_sum, rs_sd, rs_var, rs_median, rs_min, rs_max, rs_quantile
export E, Pr, rvar_rng, from_chains

include("types.jl")
include("constructors.jl")
include("abstractarray.jl")
include("broadcast.jl")
include("arithmetic.jl")
include("matmul.jl")
include("stats.jl")
include("chains.jl")

end
