# RVars.jl

[![CI](https://github.com/karimn/RVars.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/karimn/RVars.jl/actions/workflows/CI.yml)

Julia package for **sample-based random variables** — similar to R's `rvar` from the [`posterior`](https://mc-stan.org/posterior/) package. A `RVar` wraps a Monte Carlo sample (draws) behind an `AbstractArray` interface, letting you work with random quantities using normal array syntax.

## Installation

```julia
julia> # (not yet registered — add directly from GitHub)
julia> using Pkg; Pkg.add(url="https://github.com/karimn/RVars.jl")
```

## Quick Start

```julia
using RVars, Statistics

# Create a scalar random variable from 1000 draws
x = RVar(randn(1000))
mean(x)        # 0.02 (scalar)
std(x)         # 1.01 (scalar)

# Create a vector random variable (3 elements, 1000 draws each)
y = RVar(randn(1000, 3))
mean(y)        # 3-element vector
size(y)        # (3,)

# Arithmetic works element-wise across draws
z = x + y
z2 = x * 2.0
z3 = sin.(y)

# Comparisons return random booleans
gt = y .> 0
Pr(gt)         # P(y > 0) for each element

# Summary statistics
E(x)           # same as mean(x)
rs_mean(y)     # returns RVar (mean collapsed across draws)
rs_sd(y)       # returns RVar
```

## Constructors

| Input shape | Result | Draws dim |
|---|---|---|
| `Vector` `(n_draws,)` | Scalar `RVar{T,0}` | `1` |
| `Matrix` `(n_draws, n)` | Vector `RVar{T,1}` of length `n` | `1` |
| `Array{T,3}` `(n_draws, m, n)` | Matrix `RVar{T,2}` of size `(m,n)` | `1` |

### With chains

```julia
# 200 iterations × 3 chains × 5 variables
data = randn(200, 5, 3)
rd = RVar(data; with_chains=true)
nchains(rd)    # 3
niterations(rd) # 200
ndraws(rd)     # 600
```

### Constants via `as_rs`

```julia
as_rs(5.0)          # scalar constant, 1 draw
as_rs([1.0, 2.0])   # vector constant, 1 draw
as_rs(ones(3, 3))   # matrix constant, 1 draw
```

### Random generation via `rvar_rng`

```julia
rvar_rng(randn, 3)            # 3 elements, 2000 draws each (default)
rvar_rng(randn, 3; ndraws=10000)  # custom draws
```

## Broadcasting

All broadcast operations preserve the `RVar` type:

```julia
x = RVar(randn(1000, 3))
x .+ 1.0          # RVar
x .> 0            # RVar{Bool}
x .+ [1, 2, 3]    # RVar (vector RS + plain array)
```

## MCMCChains Integration

Extracting a fit gives you one random variable per model parameter, keyed by name:

```julia
using RVars, MCMCChains

chn = Chains(rand(200, 4, 5), [:a, :b, :c, :d, :e])
p = RVar(chn)     # NamedTuple: (a = RVar{Float64,0}, b = ..., ...)
p.a               # scalar RVar holding all 800 draws of :a
```

### Array parameters keep their shape

Samplers flatten an array-valued parameter into one entry per element (`x[1,1]`,
`x[2,1]`, …). Those are reassembled into a random variable of the shape the model
declared, so the draws stay hidden behind ordinary array syntax:

```julia
# a model with x[trial, patient] (2 trials × 3 patients) and a scalar sigma
(; x, sigma) = RVar(chn)

size(x)      # (2, 3) — a RVar{Float64,2}; the draws are on a hidden leading axis
x[1, 3]      # RVar{Float64,0} — every draw for trial 1, patient 3
mean(x)      # 2×3 Matrix{Float64} of posterior means
x[1, 3] - x[2, 3]   # a RVar again: uncertainty propagates draw-by-draw
```

Indices must be complete and 1-based; a gap, a duplicate, or mixed dimensionality
under one name is an error rather than a silently mangled array.

Without `MCMCChains` loaded, the same regrouping works off a plain
`(iterations, variables, chains)` array plus its parameter names:

```julia
from_chains(Array(chn), names(chn))            # NamedTuple, one entry per parameter
from_chains(Array(chn))                        # unnamed: a single vector RVar of columns
from_chains(Array(chn), names(chn); flat=true) # opt out: one named vector RVar
```

`flat=true` returns the ungrouped vector `RVar` whose elements are the flat columns,
carrying the per-element names verbatim (read them with `variables(x)`).

## API

### Draw accessors

- `draws(x)` — the raw backing array `(ndraws, dims...)`
- `draws(x; with_chains=true)` — reshape to `(iterations, chains, dims...)`
- `ndraws(x)` — total number of draws
- `nchains(x)` — number of chains
- `niterations(x)` — draws per chain

### Parameter extraction

- `RVar(chn)` / `from_chains(chn)` — a `NamedTuple` of one `RVar` per model parameter
- `rvars(x)` / `rvars(array, param_names)` — regroup per-element draws into shaped `RVar`s
- `variables(x)` — the parameter names carried by a vector `RVar`, or `nothing`

### Statistics over draws (returns plain array)

- `mean`, `std`, `var`, `median`, `minimum`, `maximum`, `quantile`

### Summary as `RVar` (returns `RVar`)

- `rs_mean`, `rs_sum`, `rs_sd`, `rs_var`, `rs_median`, `rs_min`, `rs_max`, `rs_quantile`

### Probability

- `E(x)` — alias for `mean(x)`
- `Pr(x)` — `mean(x)` for `RVar{Bool}`

## Status

Early development. See [open issues](https://github.com/karimn/RVars.jl/issues) for planned features.
