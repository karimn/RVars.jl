# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`RVars.jl` is a Julia package for treating a Monte Carlo / posterior sample as a first-class random variable. A `RVar` behaves like an ordinary `AbstractArray` of its *shape*, while carrying the underlying draws along a hidden leading axis. Arithmetic, matrix multiplication, broadcasting, and reductions all operate draw-by-draw so uncertainty propagates automatically. The design is analogous to R's `posterior`/`rvar`.

## Commands

Run from the package root.

```bash
# Full test suite — the only command that exercises the MCMCChains extension,
# because Pkg.test provides MCMCChains via [extras]/[targets]
julia --project=. -e 'using Pkg; Pkg.test()'

# Faster iteration (reuses the current environment, no sandbox). MCMCChains is
# not in this environment, so HAS_MCMCCHAINS is false and the extension
# testsets skip — expected, not a failure. Use Pkg.test to cover them.
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

If `Pkg.test()` fails with `expected package MCMCChains [c7f686f2] to be registered`, check the UUID in `Project.toml` against the registry before blaming the local registry or the sandbox. That exact error was misdiagnosed once as a compressed local registry and worked around with a throwaway environment for a whole session; the real cause was a wrong UUID in `[weakdeps]`/`[extras]`, which fails identically on every machine and in CI. The name resolves, the UUID does not — so `Pkg.add("MCMCChains")` succeeds while `Pkg.test()` fails, which is what made it look machine-specific.

## Core data model

`RVar{T, N, A}` (defined in `src/types.jl`) subtypes `AbstractArray{T, N}` but the wrapped array `A` has **N+1 dimensions**:

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

Beyond the draws and `nchains`, a `RVar` carries two independent kinds of optional metadata, and conflating them is the easiest mistake to make here:

- `names` — one `Symbol` per *element*, valid only for `N == 1` (enforced in the inner constructor). Read with `variables`.
- `dimnames` / `dimlabels` — one `Symbol` per *axis*, and optionally one label vector per axis whose length must equal that axis. Valid at any rank. Read with `dimnames`/`dimlabels`, supplied via the `dims`/`labels` keywords on `rvars`/`from_chains`/`RVar(::Chains)`, and indexed with `x[dim=label]`.

Both are dropped by operations that would leave them stale: `_maybe_names` re-checks rank and length before reattaching either, `_combine_names` and `_combine_dimmeta` decide how they merge across operands of an elementwise op. The two combine rules differ deliberately — for `names`, an operand with no names is evidence of disagreement, whereas for dimension metadata an operand carrying none simply has nothing to say (so `a .+ sigma` keeps `a`'s axes). Slicing propagates dimension metadata through `_subset_dimmeta`: an axis indexed by a scalar drops out, and a surviving axis's labels are subset by the same index applied to the data.

A vector RV (`N == 1`) may also carry an optional `names` field (`Vector{Symbol}`, one entry per element), valid only when `length(names) == size(draws, 2)`. It's supplied as an optional third positional argument to the inner/outer constructors, so untouched call sites keep producing unnamed values — names are dropped unless explicitly preserved. Read them with the exported `variables(x)` accessor. Named vector RVs support name-based indexing (`x[:alpha]`, `x[[:a, :b]]`), and ordinary subsetting (`x[2:3]`, `x[:]`, logical indexing) carries names along. Because `eltype(x) === T` while `x[i]` returns a `RVar{T, 0}`, `collect`, `Array`, `map`, `isequal`, and `hash` all need explicit methods (see "Deviations from the `AbstractArray` contract" in the `RVar` docstring).

## File layout (`src/`)

Loaded in this order by `src/RVars.jl`; later files depend on earlier ones.

- `types.jl` — the `RVar` struct, its inner constructor/invariants, and the `AbstractArray` size/axes plumbing. `_broadcast_draws` (recycle a 1-draw array up to N draws) lives here, alongside `_combine_nchains`, `_combine_names` and `_combine_dimmeta` (how nchains/names/dimension metadata combine across operands of an elementwise op) and `_maybe_names` (reattach names, dimnames and dimlabels only where they still describe the result).
- `constructors.jl` — user-facing builders: `RVar(...)`, `as_rs` (lift a constant/array to a 1-draw RV), `rvar_rng` (sample from an RNG function), `rand(RVar{T,N}, ...)`, plus the `ndraws`/`nchains`/`niterations`/`draws`/`dimnames`/`dimlabels` accessors and `_dim_index` (resolve a dimension name to an axis position).
- `abstractarray.jl` — indexing (`getindex` returns a scalar `RVar` slicing across all draws, supports name-based lookup on a named vector RV, mixed integer/colon/vector slicing that propagates dimension metadata via `_subset_dimmeta`, and keyword indexing `x[dim=label]` resolved by `_resolve_label`), `setindex!`, `reshape`, `similar`, `copy`, `isequal`/`hash`, and pretty `show` (mean ± sd, labeled with parameter names or axis labels when present).
- `broadcast.jl` — a custom `RVarStyle` broadcast style so `f.(x)` and `x .+ y` return a `RVar` instead of a plain array. `copy(::Broadcasted)` evaluates the function per logical element across all draws.
- `arithmetic.jl` — operator overloads. Two helpers do the work: `_binop_scalar` (RV ⊗ number) and `_binop_rv` (RV ⊗ RV, with draw-count and shape broadcasting). Elementwise math funcs (`sin`, `exp`, …) are metaprogrammed via an `@eval` loop.
- `matmul.jl` — batched matrix/vector products: loops over the draws axis applying `LinearAlgebra.*`/`dot` per draw. Covers RV×RV, RV×plain, and plain×RV for all rank combos.
- `stats.jl` — two distinct families:
  - **Summarise across draws** → returns *plain* numbers/arrays (the posterior summary): `mean`, `std`, `var`, `median`, `minimum`, `maximum`, `quantile`, and `E`/`Pr`. Driven by `_summarise_by_element` (reduce over axis 1 per logical element).
  - **`rs_*` (reduce-shape)** → returns a *`RVar`*, collapsing the logical shape while keeping the draws axis: `rs_mean`, `rs_sum`, `rs_sd`, `rs_var`, `rs_median`, `rs_min`, `rs_max`, `rs_quantile`.
- `params.jl` — undoes the flattening samplers apply to array-valued parameters. `_parse_param_name` splits `Symbol("x[2,3]")` into `(:x, (2, 3))`; `rvars(x)` groups a flat named vector RV into a `NamedTuple` of one `RVar` per model parameter, each with the shape its indices imply (scalar → `N=0`, `x[i]` → `N=1`, `x[i,j]` → `N=2`, …). Elements are placed by parsed index, not column order, so the result is independent of the order the sampler emitted them. Incomplete, duplicated, non-1-based, or mixed-rank index sets are errors, never silently reshaped. A name whose brackets don't parse as all-integer (`Symbol("x[a]")`) is left alone as a scalar under its full name. The `dims`/`labels` keywords attach dimension names and axis labels to the grouped results (`_attach_dims`); `labels` is keyed by *dimension* so a shared axis is declared once, and extra label keys are ignored so a whole table from `recover_types` can be passed straight in.
- `chains.jl` — `from_chains(array3d)` builds an unnamed vector RV from an `(iterations, vars, chains)` array. The string/symbol param-name overload returns the **grouped `NamedTuple`** (it routes through `rvars`), so array parameters arrive shaped; pass `flat=true` for the older behaviour — a single `RVar` carrying the per-element names, readable with `variables`, rather than a `(rv, names)` pair.

## MCMCChains integration

`MCMCChains` is a **weak dependency**. The `RVar(::MCMCChains.Chains)` and `from_chains(::Chains)` methods live in `ext/RVarsMCMCChainsExt/` and only load when the user has `MCMCChains` in scope (package extension mechanism, hence the Julia 1.9 floor). When touching chain interop, mirror the plain-array `from_chains` in `chains.jl` — note `chn.value` is indexed `(iteration, variable, chain)`, the same order the plain-array `from_chains(array)` expects.

`RVar(chn)` returns a **`NamedTuple` of one `RVar` per model parameter**, not a `RVar` — a deliberate exception to the rule that a constructor returns its own type, chosen so array parameters land in Julia already shaped. `RVar(chn; flat=true)` gives the flat named vector RV instead. Both delegate to `from_chains(arr, names; flat)`, so the grouping logic has exactly one implementation (`rvars`). Don't confuse this with the *internal* `(iterations, chains, shape...)` order that `draws(x; with_chains=true)` produces — chains and variables swap positions between the two, and reading `chn.value` with the wrong axis order was the original data-corruption bug this branch fixed.

## Conventions when extending

- Any new operation over draws must keep the draws axis as axis 1 and return a `RVar` reconstructed with the source `nchains` (see how `_binop_scalar`/`arithmetic.jl` thread `nchains` through). Combining two RVs with differing `nchains` collapses to `nchains = 1`.
- Prefer the two-helper pattern already established (a `_scalar` and a `_rv` variant) rather than open-coding each operator.
- Draw-count mismatches are only allowed when one side has exactly 1 draw (a constant), which gets recycled via `_broadcast_draws`. Otherwise error.
- Export new public names from `src/RVars.jl` and add a `@testset` to `test/runtests.jl`.
