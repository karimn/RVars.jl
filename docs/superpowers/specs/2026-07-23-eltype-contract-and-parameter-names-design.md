# RandomDraws.jl — `AbstractArray` contract and parameter names

Date: 2026-07-23
Branch: `fix/correctness-pass`

Two design decisions deferred from the correctness pass, now resolved.

## Background

`RandomDraw{T, N, A} <: AbstractArray{T, N}` declares element type `T`, but `getindex`
returns a `RandomDraw{T, 0}` holding all draws of that element. The declared element type
is therefore a lie, and generic array code that trusts it fails.

Measured on the current code, the damage is narrow. Generic fallbacks split by allocation
strategy:

- Fallbacks that allocate via `similar(A, ...)` and fill with `setindex!` work correctly,
  because both are overridden: `x[2:3]`, `x[:]`, `x[end]`, iteration, `vcat`, `hcat`,
  `sum`, `reverse`, `zero`, `similar`, `copy`.
- Fallbacks that allocate `Array{eltype(A)}` directly and `convert` each element into it
  fail: `collect(x)`, `Array(x)`, `map(f, x)`.

Separately, `from_chains(array, param_names)` returns `(rd, names)` and discards the names;
they are never stored on the value.

## Decision 1 — keep the subtype, close the holes

`RandomDraw` remains `<: AbstractArray{T, N}` with `eltype(x) === T`. The supertype is
treated as plumbing the package reuses, not a contract it promises. Dropping the subtype
was rejected: it would require reimplementing range indexing, `x[end]`, iteration,
`vcat`/`hcat`, `sum`, `reverse` and `zero` by hand. Making `eltype` honest (element type
`RandomDraw{T, 0, Vector{T}}`) was rejected: it costs `eltype(x) <: Real`, which numeric
code depends on, in exchange for fixing three methods.

### Changes

1. `Base.collect(x::RandomDraw)` returns `Array{RandomDraw{T, 0}, N}` — equivalent to
   `[x[i] for i in eachindex(x)]` reshaped to `size(x)`. Iteration already yields scalar
   `RandomDraw`s, and `collect(itr)` must agree with the comprehension.
2. `Base.Array(x::RandomDraw)` behaves identically to `collect`. In Base,
   `Array(A::AbstractArray)` is `convert(Array, A)`; agreeing with `collect` is the least
   surprising choice. Users wanting the raw `(ndraws, shape...)` store use `draws(x)`.
3. `Base.map(f, x::RandomDraw)` returns a `RandomDraw`, routed to `f.(x)`. `map(sin, x)`
   and `sin.(x)` must not disagree. Multi-arg `map(f, x, ys...)` checks that all arguments
   have equal `size` (matching `map` semantics, not `broadcast` expansion) and then
   delegates to `broadcast`.
4. `Base.isequal(x::RandomDraw, y::RandomDraw)::Bool` compares `draws`, `nchains` and
   `names`, returning a genuine `Bool`. `==` keeps its elementwise meaning
   (`src/arithmetic.jl:69` returns a `RandomDraw{Bool}` by design, mirroring R's `rvar`),
   which today leaves no Bool-returning identity test — breaking `@test x == y`, `x in xs`
   and `Dict` keys.
5. `Base.hash(x::RandomDraw, h::UInt)` hashes the same three fields, so `isequal` implies
   equal hashes and `Dict` keys actually work. This is required, not optional: Base's
   generic `hash(::AbstractArray)` hashes elements, an element of a `RandomDraw` is
   another `RandomDraw`, and for `N == 0` that is itself — so `hash` throws
   `StackOverflowError` on any `RandomDraw` without this method. A pre-existing latent
   bug, surfaced by adding `isequal`.
6. The `RandomDraw` docstring gains an explicit "Deviations from the `AbstractArray`
   contract" section covering the `eltype` mismatch and the comparison operators.

## Decision 2 — store parameter names on the value

### Storage

```julia
struct RandomDraw{T, N, A <: AbstractArray{T}} <: AbstractArray{T, N}
    draws::A
    nchains::Int
    names::Union{Nothing, Vector{Symbol}}
end
```

The inner constructor takes `names` as an optional third positional argument defaulting to
`nothing`. Every existing `RandomDraw{T, N, typeof(d)}(d, nc)` call site compiles unchanged
and keeps producing unnamed values. This makes "drop names unless explicitly preserved" the
compiler's default rather than a rule enforced by discipline at ~30 call sites.

Invariant, checked in the inner constructor:

```
names === nothing || (N == 1 && length(names) == size(draws, 2))
```

Names are a vector-RV concept. `N = 0` has no elements to label; `N ≥ 2` would need one
name vector per axis (the `NamedDims.jl` / `DimensionalData.jl` problem) with its own
propagation rules under `permutedims` and matrix multiplication. Passing names with
`N ≠ 1` is an error, not a silently chosen interpretation.

This costs nothing in practice: `from_chains` always produces `N = 1`. Samplers flatten
matrix parameters into a flat parameter vector with names like `Symbol("beta[1,1]")`.

### Accessor

`variables(x)` returns `Union{Nothing, Vector{Symbol}}`. Exported. Named after
`posterior::variables`, the R function this package is modelled on. Not `Base.names`,
which means something else in Julia.

### Attaching names

- `from_chains(array, param_names)` returns a single `RandomDraw` carrying the names,
  replacing the `(rd, names)` tuple. **Breaking change** — acceptable at v0.1.0, and the
  point of the exercise.
- `RandomDraw(x::AbstractArray; nchains, with_chains, names=nothing)` gains the kwarg.
- The MCMCChains extension attaches names automatically. `names(chn)` returns a
  `Vector{Symbol}` in the same order as the `:var` axis of `chn.value`, so
  `RandomDraw(chn)` and `from_chains(chn)` gain named access with no extra bookkeeping.
  This is the highest-value part of the feature: `Chains` always carries names and they
  are currently discarded.

### Propagation — default drop, opt-in preserve

| Operation | Names |
|---|---|
| `x[:alpha]`, `x[3]` → scalar RV | dropped (`N = 0`) |
| `x[2:3]`, `x[[:a, :b]]`, `x[:]` | subset |
| `x .+ 1`, `sin(x)`, `-x` | preserved |
| `x .+ y`, both named, names agree | preserved |
| `x .+ y`, `y` a single-draw constant | preserved from `x` |
| `x .+ y`, names differ, or `y` is multi-draw and unnamed | dropped |
| `copy(x)` | preserved |
| `similar`, `reshape`, `vcat`, matmul, `rs_*` | dropped |
| `mean(x)`, `quantile(x)`, `E`, `Pr` | not applicable — already return plain arrays |

`similar` dropping while `copy` preserves is intentional: `similar` returns uninitialised
storage, so element identity has not been carried over.

Implemented by `_combine_names(operands...)` in `src/types.jl`, deliberately mirroring the
existing `_combine_nchains`: single-draw operands are constants and defer to the others;
two disagreeing name sets cannot be reconciled, so the result is unnamed. Applied in
`_binop_scalar`, `_binop_rv`, the unary and metaprogrammed elementwise math functions, and
broadcast's `similar(::Broadcasted, ::Type)`.

### Name-based indexing

- `x[:alpha]` → scalar `RandomDraw{T, 0}`.
- `x[[:alpha, :beta]]` → `RandomDraw{T, 1}` carrying those two names.

Both resolve symbols to integer positions and delegate to the integer methods. An unknown
name raises an error listing the available names.

This requires two new explicit methods, because the generic fallbacks for both route
through `similar`, which by design cannot know the names:

- `Base.getindex(x::RandomDraw{T, 1}, I::AbstractVector{<:Integer})` — subsets names by the
  same indices. Slicing `_flat_store` columns directly also avoids the fallback's
  element-by-element loop.
- `Base.getindex(x::RandomDraw{T, 1}, ::Colon)` — delegates to the above with
  `1:length(x)`, so `x[:]` keeps all names rather than silently dropping them.

`Symbol` and `Colon` are distinct types, so `x[:alpha]` and `x[:]` never collide.

### Display

When names are present, `show` prints `[alpha] 1.23 ± 0.45` in place of `[1] 1.23 ± 0.45`.

## Testing

Four new `@testset` blocks in `test/runtests.jl`, all value-based:

1. `collect` / `Array` / `map` / `isequal` — including that `map(sin, x)` equals `sin.(x)`
   and that `isequal` returns a `Bool` where `==` returns a `RandomDraw{Bool}`.
2. Name propagation — one assertion per row of the table above.
3. Name-based indexing — scalar and vector forms, correct subsetting of names, and the
   unknown-name error.
4. MCMCChains names round-trip, guarded by the existing `HAS_MCMCCHAINS` flag.

## Out of scope

- Per-axis dimnames for `N ≥ 2`.
- Preserving names through `vcat`.
- Returning named results from the summarise family (`mean`, `quantile`, `E`, `Pr`).
- Inverting the design so a scalar RV subtypes `Number` (the `MonteCarloMeasurements.jl`
  approach). Noted as a road not taken; it would be a rewrite, not a change.
