# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`RandomDraws.jl` is a Julia package for treating a Monte Carlo / posterior sample as a first-class random variable. A `RandomDraw` behaves like an ordinary `AbstractArray` of its *shape*, while carrying the underlying draws along a hidden leading axis. Arithmetic, matrix multiplication, broadcasting, and reductions all operate draw-by-draw so uncertainty propagates automatically. The design is analogous to R's `posterior`/`rvar`.

## Commands

Run from the package root.

```bash
# Full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Faster iteration (reuses the current environment, no sandbox)
julia --project=. test/runtests.jl

# Run one @testset — Julia has no CLI test filter, so temporarily wrap the
# suite or use TestItems is NOT set up here; instead comment out other
# @testset blocks in test/runtests.jl, or from the REPL:
julia --project=.
# then: include("test/runtests.jl")

# Instantiate deps after cloning
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

CI (`.github/workflows/CI.yml`) runs `Pkg.test()` on Julia 1.9, 1.10, and 1.11 for every PR and push to `main`. The `compat` floor is Julia 1.9 — avoid syntax/stdlib newer than that.

## Core data model

`RandomDraw{T, N, A}` (defined in `src/types.jl`) subtypes `AbstractArray{T, N}` but the wrapped array `A` has **N+1 dimensions**:

- **Axis 1 = the draws axis** (Monte Carlo samples). Always present.
- **Axes 2…N+1 = the logical shape** the user sees via `size(x)`.

So `N` is the dimensionality the *user* observes, not the stored array:

| User sees        | `N` | Wrapped array `A`         |
|------------------|-----|---------------------------|
| scalar RV        | 0   | vector `(ndraws,)`        |
| vector RV        | 1   | matrix `(ndraws, k)`      |
| matrix RV        | 2   | 3-tensor `(ndraws, m, n)` |

This is why almost every method peels the first axis: `size(d, 1)` is the draw count, `size(d)[2:end]` is the visible shape. When adding methods, preserve this invariant — the constructor enforces `ndims(draws) == N + 1`.

**Chains** are packed *into* axis 1: `ndraws == niterations * nchains`, stored contiguously. `nchains` is a plain field; `niterations(x) = ndraws ÷ nchains`. Chains are only materialized as a separate axis on demand via `draws(x; with_chains=true)`, which reshapes to `(niterations, nchains, shape...)`. The divisibility check (`nchains` must divide `ndraws`) lives in the inner constructor.

A vector RV (`N == 1`) may also carry an optional `names` field (`Vector{Symbol}`, one entry per element), valid only when `length(names) == size(draws, 2)`. It's supplied as an optional third positional argument to the inner/outer constructors, so untouched call sites keep producing unnamed values — names are dropped unless explicitly preserved. Read them with the exported `variables(x)` accessor. Named vector RVs support name-based indexing (`x[:alpha]`, `x[[:a, :b]]`), and ordinary subsetting (`x[2:3]`, `x[:]`, logical indexing) carries names along. Because `eltype(x) === T` while `x[i]` returns a `RandomDraw{T, 0}`, `collect`, `Array`, `map`, `isequal`, and `hash` all need explicit methods (see "Deviations from the `AbstractArray` contract" in the `RandomDraw` docstring).

## File layout (`src/`)

Loaded in this order by `src/RandomDraws.jl`; later files depend on earlier ones.

- `types.jl` — the `RandomDraw` struct, its inner constructor/invariants, and the `AbstractArray` size/axes plumbing. `_broadcast_draws` (recycle a 1-draw array up to N draws) lives here, alongside `_combine_nchains` and `_combine_names` (how nchains/names combine across operands of an elementwise op) and `_maybe_names` (attach names only when they still match the result's length).
- `constructors.jl` — user-facing builders: `RandomDraw(...)`, `as_rs` (lift a constant/array to a 1-draw RV), `rvar_rng` (sample from an RNG function), `rand(RandomDraw{T,N}, ...)`, plus the `ndraws`/`nchains`/`niterations`/`draws` accessors.
- `abstractarray.jl` — indexing (`getindex` returns a scalar `RandomDraw` slicing across all draws, and also supports name-based lookup on a named vector RV), `setindex!`, `reshape`, `similar`, `copy`, `isequal`/`hash`, and pretty `show` (mean ± sd, labeled with parameter names when present).
- `broadcast.jl` — a custom `RandomDrawStyle` broadcast style so `f.(x)` and `x .+ y` return a `RandomDraw` instead of a plain array. `copy(::Broadcasted)` evaluates the function per logical element across all draws.
- `arithmetic.jl` — operator overloads. Two helpers do the work: `_binop_scalar` (RV ⊗ number) and `_binop_rv` (RV ⊗ RV, with draw-count and shape broadcasting). Elementwise math funcs (`sin`, `exp`, …) are metaprogrammed via an `@eval` loop.
- `matmul.jl` — batched matrix/vector products: loops over the draws axis applying `LinearAlgebra.*`/`dot` per draw. Covers RV×RV, RV×plain, and plain×RV for all rank combos.
- `stats.jl` — two distinct families:
  - **Summarise across draws** → returns *plain* numbers/arrays (the posterior summary): `mean`, `std`, `var`, `median`, `minimum`, `maximum`, `quantile`, and `E`/`Pr`. Driven by `_summarise_by_element` (reduce over axis 1 per logical element).
  - **`rs_*` (reduce-shape)** → returns a *`RandomDraw`*, collapsing the logical shape while keeping the draws axis: `rs_mean`, `rs_sum`, `rs_sd`, `rs_var`, `rs_median`, `rs_min`, `rs_max`, `rs_quantile`.
- `chains.jl` — `from_chains(array3d)` builds an RV from an `(iterations, vars, chains)` array; the string/symbol param-name overload returns a single `RandomDraw` carrying the names (readable with `variables`), not a `(rv, names)` pair.

## MCMCChains integration

`MCMCChains` is a **weak dependency**. The `RandomDraw(::MCMCChains.Chains)` and `from_chains(::Chains)` methods live in `ext/RandomDrawsMCMCChainsExt/` and only load when the user has `MCMCChains` in scope (package extension mechanism, hence the Julia 1.9 floor). When touching chain interop, mirror the plain-array `from_chains` in `chains.jl` — note `chn.value` is indexed `(iteration, variable, chain)`, the same order the plain-array `from_chains(array)` expects. Don't confuse this with the *internal* `(iterations, chains, shape...)` order that `draws(x; with_chains=true)` produces — chains and variables swap positions between the two, and reading `chn.value` with the wrong axis order was the original data-corruption bug this branch fixed.

## Conventions when extending

- Any new operation over draws must keep the draws axis as axis 1 and return a `RandomDraw` reconstructed with the source `nchains` (see how `_binop_scalar`/`arithmetic.jl` thread `nchains` through). Combining two RVs with differing `nchains` collapses to `nchains = 1`.
- Prefer the two-helper pattern already established (a `_scalar` and a `_rv` variant) rather than open-coding each operator.
- Draw-count mismatches are only allowed when one side has exactly 1 draw (a constant), which gets recycled via `_broadcast_draws`. Otherwise error.
- Export new public names from `src/RandomDraws.jl` and add a `@testset` to `test/runtests.jl`.
