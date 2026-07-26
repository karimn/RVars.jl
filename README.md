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

### Naming the dimensions

A chain records `a[1,2]` and nothing about what those positions *mean* — that lives in
your model, not the fit. Declare it at extraction, in the spirit of tidybayes'
`spread_draws(fit, a[trial, arm])`:

```julia
p = RVar(chn; dims   = (a = (:trial, :arm), b = (:trial, :arm, :time)),
              labels = (arm = ["control", "drug", "placebo"],))

dimnames(p.a)          # (:trial, :arm)
dimlabels(p.a, :arm)   # ["control", "drug", "placebo"]
```

Labels are keyed by *dimension*, not by parameter, so `:arm` means the same thing
everywhere it appears and is declared once. Dimensions can then be indexed by name, and
positions by label:

```julia
p.a[trial=1, arm=:drug]        # scalar RVar
p.a[arm=:drug]                 # RVar over trials; :arm drops out
p.a[arm=["drug", "placebo"]]   # RVar{Float64,2}, :arm relabelled to the subset
p.a[trial=1, arm=2]            # positions still work on a labelled axis
```

Omitted dimensions default to `:`. Metadata follows the value through slicing and
shape-preserving arithmetic (`p.a .+ 1`, `sin.(p.a)`, `p.a .+ p.sigma` all keep
`(:trial, :arm)`), and is dropped rather than left stale by anything that changes the
rank. It also shows up when printing:

```
julia> p.a[trial=1]
RVar{Float64}<200,2>[arm=3] mean ± sd:
[control] 0.04 ± 0.92
[drug] 0.02 ± 0.99
[placebo] 0.11 ± 0.94
```

### Recovering labels from your data

Rather than typing the levels out, derive them from the data the model was fitted to —
the analogue of tidybayes' `recover_types`:

```julia
labs = recover_types(df)     # or recover_types(eachcol(df)) for a DataFrame
p = RVar(chn; dims = (a = (:trial, :arm),), labels = labs)
```

Every non-continuous column becomes one entry; columns that aren't dimensions of the fit
are ignored, so passing a whole table is fine. Note that the *order* of recovered labels
must match the integer coding your model used — `recover_types` sorts by default, matching
how R factors and `CategoricalArray`s number their levels, and `sorted=false` keeps
first-appearance order. If your model indexed some other way, pass `labels` explicitly.

The same regrouping works off a plain `(iterations, variables, chains)` array plus its
parameter names, with no `MCMCChains` dependency:

```julia
arr = randn(200, 7, 4)   # 200 iterations × 7 variables × 4 chains
pnames = [Symbol("x[1,1]"), Symbol("x[2,1]"), Symbol("x[1,2]"),
          Symbol("x[2,2]"), Symbol("x[1,3]"), Symbol("x[2,3]"), :sigma]

from_chains(arr, pnames)             # NamedTuple: (x = RVar{Float64,2}, sigma = ...)
from_chains(arr)                     # unnamed: a single vector RVar, one element per column
from_chains(arr, pnames; flat=true)  # opt out of grouping: one named vector RVar
```

(To get such an array out of a `Chains` object, reach for `Array(chn.value)` — plain
`Array(chn)` flattens iterations and chains together into a matrix.)

`flat=true` returns the ungrouped vector `RVar` whose elements are the flat columns,
carrying the per-element names verbatim (read them with `variables(x)`).

### Long tables via `gather_draws`

Downstream consumers — `DataFrame`, `CSV.write`, plotting via
[AlgebraOfGraphics](https://github.com/MakieOrg/AlgebraOfGraphics.jl) — are table-shaped,
not array-shaped. `gather_draws` is the tidybayes `gather_draws` analogue: it flattens a
`RVar` (or a `NamedTuple` of them) into one row per draw × element, with one column per
dimension carrying that dimension's labels. The result is a `NamedTuple` of equal-length
vectors, which is already a valid Tables.jl column table — no `Tables.jl` dependency
needed.

```julia
p = RVar(chn; dims = (a = (:trial, :arm),), labels = (arm = ["control", "drug"],))

gather_draws(p.a)
# (trial = [...], arm = [...], chain = [...], draw = [...], value = [...])

gather_draws(p)
# (variable = [...], trial = [...], arm = [...], chain = [...], draw = [...], value = [...])

using AlgebraOfGraphics, CairoMakie
data(gather_draws(p.a)) * mapping(:arm, :value; color = :arm) * visual(BoxPlot) |> draw
```

Gathering several parameters at once takes the union of their dimension columns, filling
`missing` where a parameter lacks a dimension the others have. Label element types are
preserved rather than stringified, so an `Int` or `Symbol` axis label round-trips as such.

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

### Dimensions

- `dims` / `labels` keywords on `RVar(chn)`, `from_chains`, `rvars` — name the axes
- `dimnames(x)` — the names of `x`'s axes, or `nothing`
- `dimlabels(x)` / `dimlabels(x, dim)` — the labels along each axis
- `x[dim=index]` — index by dimension name, by position or by label
- `recover_types(data)` — derive labels from the data the model was fitted to
- `gather_draws(x)` / `gather_draws(nt)` — flatten a `RVar` (or `NamedTuple` of them) into a long table

### Statistics over draws (returns plain array)

- `mean`, `std`, `var`, `median`, `minimum`, `maximum`, `quantile`

### Summary as `RVar` (returns `RVar`)

- `rs_mean`, `rs_sum`, `rs_sd`, `rs_var`, `rs_median`, `rs_min`, `rs_max`, `rs_quantile`

### Probability

- `E(x)` — alias for `mean(x)`
- `Pr(x)` — `mean(x)` for `RVar{Bool}`

## Status

Early development. See [open issues](https://github.com/karimn/RVars.jl/issues) for planned features.
