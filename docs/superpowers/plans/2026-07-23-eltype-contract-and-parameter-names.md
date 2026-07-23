# `AbstractArray` Contract and Parameter Names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three `AbstractArray` contract holes in `RandomDraws.jl` (`collect`, `Array`, `map`), and let a vector random variable carry parameter names that survive elementwise operations and index subsetting.

**Architecture:** `RandomDraw` keeps subtyping `AbstractArray{T, N}`; the three failing generic fallbacks get explicit methods. Names live in a new optional third struct field, defaulted to `nothing`, so all existing construction sites keep producing unnamed values and "drop names unless explicitly preserved" is the compiler's default. Two helpers in `src/types.jl` — `_combine_names` (mirroring the existing `_combine_nchains`) and `_maybe_names` (attach only where the names still describe the result) — are the single funnel through which names propagate.

**Tech Stack:** Julia (stdlib only: `LinearAlgebra`, `Statistics`, `Random`); `MCMCChains` as a weak dependency loaded via a package extension; `Test` for the suite.

**Spec:** `docs/superpowers/specs/2026-07-23-eltype-contract-and-parameter-names-design.md`

## Global Constraints

- Julia compat floor is `1.9` (`Project.toml`). Do not use syntax or stdlib newer than 1.9. The floor exists because package extensions require 1.9.
- No new package dependencies. `MCMCChains` stays a weak dependency declared under `[weakdeps]`/`[extensions]` and a test dependency under `[extras]`/`[targets]`.
- The core invariant: the wrapped array has `N + 1` dimensions, axis 1 is the draws axis, axes `2:N+1` are the visible shape. Every new method must preserve it.
- Chains are packed into axis 1 with iterations fastest. Any reconstruction must thread the source `nchains` through.
- New public names must be added to the `export` list in `src/RandomDraws.jl`.
- Parameter names are valid **only** for `N == 1` and must have `length(names) == size(draws, 2)`.
- **Running tests:** `julia --project=. test/runtests.jl` from the package root. Julia has no CLI test filter, so every "run the test" step runs the whole suite; check the named `@testset` in the summary output. Do **not** use `Pkg.test()` — it fails on this machine because the local General registry is compressed (`General.tar.gz`) and the sandbox resolve cannot find `MCMCChains`. Under the plain command, `HAS_MCMCCHAINS` is `false` and the extension testsets skip; that is expected.
- New testsets go inside the outer `@testset "RandomDraws.jl"` block, inserted immediately **before** the `if HAS_MCMCCHAINS` guard at `test/runtests.jl:491`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/types.jl` | Struct, invariants, `_broadcast_draws`, `_combine_nchains` | Add `names` field + validation; add `_combine_names`, `_maybe_names`; extend docstring |
| `src/constructors.jl` | User-facing builders and accessors | Add `variables`; add `names` kwarg to `RandomDraw(::AbstractArray; ...)` |
| `src/abstractarray.jl` | Indexing, `similar`, `copy`, `show` | Add `collect`/`Array`/`map`/`isequal`; add integer-vector, `Colon`, `Bool`-vector and `Symbol` indexing; preserve names in `copy`; label `show` with names |
| `src/arithmetic.jl` | Operator overloads | Thread `_maybe_names` through `_binop_scalar`, `_binop_rv`, unary ops, the `@eval` math loop |
| `src/broadcast.jl` | `RandomDrawStyle`, `similar`, `copy` | Thread `_combine_names`/`_maybe_names` through `similar(::Broadcasted, ::Type)` |
| `src/chains.jl` | `from_chains` | Return a single named `RandomDraw` instead of a tuple |
| `src/RandomDraws.jl` | Module, includes, exports | Export `variables` |
| `ext/RandomDrawsMCMCChainsExt/RandomDrawsMCMCChainsExt.jl` | `Chains` interop | Attach `names(chn)` automatically |
| `test/runtests.jl` | Test suite | Update the tuple assertions at lines 181–187; add five testsets |

Tasks run in order: Task 1 is independent; Task 2 must precede 3–7 (they all need the field); Task 3 must precede 4–6 (they need a way to build a named value).

---

### Task 1: Close the `collect` / `Array` / `map` holes

**Files:**
- Modify: `src/abstractarray.jl` (append after `Base.copy`, before `Base.show`)
- Test: `test/runtests.jl` (insert before the `if HAS_MCMCCHAINS` guard at line 491)

**Interfaces:**
- Consumes: `_flat_store`, `Base.getindex(::RandomDraw, ::Int)` (already present in `src/abstractarray.jl`).
- Produces: `Base.collect(::RandomDraw)`, `Base.Array(::RandomDraw)`, `Base.map(f, ::RandomDraw)`, `Base.map(f, ::RandomDraw, ys...)`. No later task depends on these.

- [ ] **Step 1: Write the failing test**

Insert into `test/runtests.jl` immediately before the `if HAS_MCMCCHAINS` line:

```julia
    @testset "AbstractArray contract holes (collect/Array/map)" begin
        d = reshape(collect(1.0:12.0), 4, 3)   # 4 draws, length-3 vector RV
        x = RandomDraw(d)

        c = collect(x)
        @test c isa Vector
        @test size(c) == (3,)
        @test all(e -> e isa RandomDraw, c)
        # collect must agree with the comprehension, element for element.
        for j in 1:3
            @test draws(c[j]) == d[:, j]
        end

        a = Array(x)
        @test size(a) == (3,)
        for j in 1:3
            @test draws(a[j]) == d[:, j]
        end

        # An N=2 RV collects to a matrix of scalar RVs, preserving position.
        d2 = reshape(collect(1.0:24.0), 4, 2, 3)
        y = RandomDraw(d2)
        c2 = collect(y)
        @test size(c2) == (2, 3)
        for i in 1:2, j in 1:3
            @test draws(c2[i, j]) == d2[:, i, j]
        end

        # map and broadcast must not disagree.
        m = map(sin, x)
        @test m isa RandomDraw
        @test draws(m) ≈ sin.(d)
        @test draws(m) ≈ draws(sin.(x))

        # Multi-argument map.
        x2 = RandomDraw(d .* 2)
        m2 = map(+, x, x2)
        @test m2 isa RandomDraw
        @test draws(m2) ≈ d .+ (d .* 2)

        # map requires equal sizes; it must not silently broadcast-expand.
        z = RandomDraw(reshape(collect(1.0:8.0), 4, 2))
        @test_throws DimensionMismatch map(+, x, z)
    end

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL. The `AbstractArray contract holes (collect/Array/map)` testset errors on the first `collect(x)` with `MethodError: Cannot convert an object of type RandomDraw{Float64, 0, Vector{Float64}} to an object of type Float64`.

- [ ] **Step 3: Write the implementation**

Append to `src/abstractarray.jl`, after the `Base.copy` method and before `Base.show`:

```julia
# `collect`, `Array` and `map` allocate `Array{eltype(x)}` up front and convert each
# element into it. That is the one place the `eltype(x) === T` declaration cannot hold:
# an element of a RandomDraw is a scalar RandomDraw, not a T. Materialise the scalar RVs
# explicitly. (The similar-based fallbacks — x[2:3], vcat, reverse, sum — need no help,
# because `similar` is overridden to return a RandomDraw.)
function Base.collect(x::RandomDraw{T, N}) where {T, N}
    out = [x[i] for i in eachindex(x)]
    return reshape(out, size(x))
end

Base.Array(x::RandomDraw) = collect(x)

# `map(sin, x)` and `sin.(x)` must agree, so route map through broadcast. Unlike
# broadcast, map does not expand singleton dimensions, so check shapes first.
Base.map(f, x::RandomDraw) = broadcast(f, x)

function Base.map(f, x::RandomDraw, ys...)
    for y in ys
        size(y) == size(x) || throw(DimensionMismatch(
            "map requires equal sizes, got $(size(x)) and $(size(y))"))
    end
    return broadcast(f, x, ys...)
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. The suite summary shows `AbstractArray contract holes (collect/Array/map)` with 0 failures and 0 errors, and no previously passing testset regresses.

- [ ] **Step 5: Commit**

```bash
git add src/abstractarray.jl test/runtests.jl
git commit -m "Close the collect/Array/map holes in the AbstractArray contract"
```

---

### Task 2: Add the `names` field, `variables`, and `isequal`

**Files:**
- Modify: `src/types.jl:27-44` (the struct and inner constructor), and the docstring at `src/types.jl:1-26`
- Modify: `src/constructors.jl` (append `variables` after the `niterations` accessor at line 20)
- Modify: `src/abstractarray.jl` (append `isequal` after the `map` methods from Task 1)
- Modify: `src/RandomDraws.jl:7` (export list)
- Test: `test/runtests.jl` (insert before the `if HAS_MCMCCHAINS` guard)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - Struct field `names::Union{Nothing, Vector{Symbol}}`, third positional argument of the inner constructor `RandomDraw{T,N,A}(draws, nchains=1, names=nothing)`.
  - `variables(x::RandomDraw) -> Union{Nothing, Vector{Symbol}}`, exported.
  - `Base.isequal(x::RandomDraw, y::RandomDraw) -> Bool`.

- [ ] **Step 1: Write the failing test**

Insert into `test/runtests.jl` immediately before the `if HAS_MCMCCHAINS` line:

```julia
    @testset "Names field, variables accessor, isequal" begin
        d = reshape(collect(1.0:12.0), 4, 3)

        # Default: no names, and existing 2-argument construction still works.
        x = RandomDraw(d)
        @test variables(x) === nothing
        @test RandomDraw{Float64, 1, typeof(d)}(d, 1) isa RandomDraw

        # Names attach via the inner constructor's third positional argument.
        named = RandomDraw{Float64, 1, typeof(d)}(d, 1, [:a, :b, :c])
        @test variables(named) == [:a, :b, :c]
        @test draws(named) == d
        @test nchains(named) == 1

        # Wrong number of names is rejected.
        @test_throws ErrorException RandomDraw{Float64, 1, typeof(d)}(d, 1, [:a, :b])

        # Names are only valid for N == 1.
        d0 = collect(1.0:4.0)
        @test_throws ErrorException RandomDraw{Float64, 0, typeof(d0)}(d0, 1, [:a])
        d2 = reshape(collect(1.0:24.0), 4, 2, 3)
        @test_throws ErrorException RandomDraw{Float64, 2, typeof(d2)}(d2, 1, [:a, :b])

        # isequal returns a genuine Bool, unlike == which is elementwise by design.
        @test isequal(x, RandomDraw(d)) === true
        @test (x == RandomDraw(d)) isa RandomDraw
        @test isequal(x, named) === false                       # names differ
        @test isequal(x, RandomDraw(d, nchains=2)) === false     # nchains differ
        @test isequal(x, RandomDraw(d .+ 1)) === false           # draws differ

        # isequal makes RandomDraw usable as a Dict key.
        dict = Dict(x => "unnamed", named => "named")
        @test dict[RandomDraw(d)] == "unnamed"
        @test length(dict) == 2
    end

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL. The `Names field, variables accessor, isequal` testset errors with `UndefVarError: variables not defined`.

- [ ] **Step 3: Write the implementation**

Replace the struct definition at `src/types.jl:27-44` with:

```julia
struct RandomDraw{T, N, A <: AbstractArray{T}} <: AbstractArray{T, N}
    draws::A
    nchains::Int
    names::Union{Nothing, Vector{Symbol}}

    function RandomDraw{T, N, A}(draws::A, nchains::Int=1,
                                 names::Union{Nothing, AbstractVector{Symbol}}=nothing
                                 ) where {T, N, A <: AbstractArray{T}}
        nchains >= 1 || error("nchains must be >= 1")
        nd = size(draws, 1)
        if nd % nchains != 0
            error("Number of chains ($nchains) does not divide number of draws ($nd)")
        end
        expected_dims = N + 1
        actual_dims = ndims(draws)
        if actual_dims != expected_dims
            error("Expected draws array with $expected_dims dimensions (draws × shape), got $actual_dims")
        end
        if names !== nothing
            # Names label the elements of a single logical axis. N == 0 has no elements to
            # label; N >= 2 would need one name vector per axis, a different feature.
            N == 1 || error("Parameter names are only supported for vector random variables (N == 1), got N == $N")
            length(names) == size(draws, 2) ||
                error("Got $(length(names)) names for a length-$(size(draws, 2)) random variable")
        end
        new{T, N, A}(draws, nchains, names === nothing ? nothing : collect(Symbol, names))
    end
end
```

Append this section to the `RandomDraw` docstring at `src/types.jl`, immediately before the closing `"""` on line 26:

```
A vector random variable (`N == 1`) may additionally carry parameter names; see
[`variables`](@ref) and [`from_chains`](@ref). Names are dropped by any operation that
does not preserve elementwise identity.

# Deviations from the `AbstractArray` contract

`RandomDraw` subtypes `AbstractArray` to reuse Julia's indexing and `similar` plumbing,
not to promise the full contract. Two deliberate deviations:

- `eltype(x) === T`, but `x[i]` returns a `RandomDraw{T, 0}` — every element is itself a
  random variable. `collect`, `Array` and `map` have explicit methods because of this;
  everything routed through `similar` works unchanged. Use [`draws`](@ref) for the raw
  `(ndraws, shape...)` store.
- `==`, `<`, `<=`, `>`, `>=` compare draw-by-draw and return a `RandomDraw{Bool}`, not a
  `Bool` (matching R's `rvar`). Use `isequal` for a `Bool`-returning identity test, which
  is what `Dict` and `in` need.
```

Append to `src/constructors.jl`, after the `niterations` accessor (line 20):

```julia
"""
    variables(x)

The parameter names carried by `x`, or `nothing` if it has none. Only vector random
variables (`N == 1`) can carry names; see [`from_chains`](@ref).
"""
variables(x::RandomDraw) = x.names
```

Append to `src/abstractarray.jl`, after the `map` methods from Task 1:

```julia
# `==` is elementwise by design (src/arithmetic.jl), so it cannot answer "are these the
# same random variable?". `isequal` does, and is what Dict and `in` dispatch on.
function Base.isequal(x::RandomDraw, y::RandomDraw)
    return isequal(x.draws, y.draws) && x.nchains == y.nchains && isequal(x.names, y.names)
end
```

Change the export line at `src/RandomDraws.jl:7` to:

```julia
export RandomDraw, as_rs, draws, ndraws, nchains, niterations, variables
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. `Names field, variables accessor, isequal` reports 0 failures, and every pre-existing testset still passes — the new field is optional, so the roughly 30 two-argument construction sites are unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/constructors.jl src/abstractarray.jl src/RandomDraws.jl test/runtests.jl
git commit -m "Add optional parameter-names field, variables accessor, and isequal"
```

---

### Task 3: Attach names via `RandomDraw` and `from_chains`

**Files:**
- Modify: `src/constructors.jl:58-79` (the `RandomDraw(::AbstractArray; ...)` method and its docstring at lines 47-57)
- Modify: `src/chains.jl` (the docstring at lines 1-16 and the two named overloads at lines 27-34)
- Modify: `test/runtests.jl:181-187` (assertions that destructure the old tuple return)
- Test: `test/runtests.jl` (insert before the `if HAS_MCMCCHAINS` guard)

**Interfaces:**
- Consumes: the `names` field and `variables` from Task 2.
- Produces:
  - `RandomDraw(x::AbstractArray; nchains=1, with_chains=false, names=nothing)`.
  - `from_chains(array, param_names) -> RandomDraw{T, 1}` carrying the names. **This replaces the previous `(rd, names)` tuple return.**

- [ ] **Step 1: Write the failing test**

First, replace `test/runtests.jl:181-187` (inside `@testset "from_chains (MCMCChains interop)"`) — the old tuple-destructuring assertions — with:

```julia
        rd2 = from_chains(data, [:mu, :sigma, :alpha, :beta, :lp])
        @test rd2 isa RandomDraw
        @test variables(rd2) == [:mu, :sigma, :alpha, :beta, :lp]

        rd3 = from_chains(data, ["mu", "sigma", "alpha", "beta", "lp"])
        @test rd3 isa RandomDraw
        @test variables(rd3) == [:mu, :sigma, :alpha, :beta, :lp]
```

Then insert this testset immediately before the `if HAS_MCMCCHAINS` line:

```julia
    @testset "Attaching names" begin
        # from_chains attaches names to the value itself, not alongside it.
        A = [100c + 10v + i for i in 1:2, v in 1:3, c in 1:4]
        rd = from_chains(A, [:alpha, :beta, :gamma])
        @test variables(rd) == [:alpha, :beta, :gamma]
        @test nchains(rd) == 4
        # Attaching names must not disturb the draws layout.
        @test draws(rd) == draws(from_chains(A))

        # Wrong number of names is rejected.
        @test_throws ErrorException from_chains(A, [:alpha, :beta])

        # The RandomDraw constructor takes a names kwarg.
        d = reshape(collect(1.0:12.0), 4, 3)
        x = RandomDraw(d; names=[:a, :b, :c])
        @test variables(x) == [:a, :b, :c]
        @test draws(x) == d

        # names composes with nchains.
        xc = RandomDraw(d; nchains=2, names=[:a, :b, :c])
        @test nchains(xc) == 2
        @test variables(xc) == [:a, :b, :c]

        # names composes with with_chains.
        wc = reshape(collect(1.0:24.0), 2, 4, 3)   # (iterations, chains, 3 vars)
        xw = RandomDraw(wc; with_chains=true, names=[:a, :b, :c])
        @test nchains(xw) == 4
        @test variables(xw) == [:a, :b, :c]

        # Names on a non-vector RV are rejected.
        d2 = reshape(collect(1.0:24.0), 4, 2, 3)
        @test_throws ErrorException RandomDraw(d2; names=[:a, :b])
    end

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL. `from_chains (MCMCChains interop)` fails on `variables(rd2)` returning `nothing` (the overload still discards the names), and `Attaching names` errors with `MethodError: no method matching RandomDraw(::Matrix{Float64}; names::Vector{Symbol})`.

- [ ] **Step 3: Write the implementation**

Replace the `RandomDraw(::AbstractArray; ...)` method at `src/constructors.jl:58-79` with:

```julia
function RandomDraw(x::AbstractArray{T}; nchains::Int=1, with_chains::Bool=false,
                    names::Union{Nothing, AbstractVector{Symbol}}=nothing) where {T}
    if with_chains
        if nchains != 1
            @warn "with_chains=true derives nchains from the data; ignoring nchains=$nchains"
        end
        sz = size(x)
        length(sz) >= 2 || error("with_chains=true requires >= 2 dims (iterations, chains, ...)")
        n_iter, n_chain = sz[1], sz[2]
        rest = sz[3:end]
        new_sz = (n_iter * n_chain, rest...)
        reshaped = reshape(x, new_sz)
        n_out = length(rest)
        return RandomDraw{T, n_out, typeof(reshaped)}(reshaped, n_chain, names)
    end
    if ndims(x) == 1
        # A flat vector is a scalar RV whose draws are the whole vector; honor nchains.
        return RandomDraw{T, 0, typeof(x)}(x, nchains, names)
    end
    sz = size(x)
    n_out = length(sz) - 1
    return RandomDraw{T, n_out, typeof(x)}(x, nchains, names)
end
```

Add this line to that method's docstring at `src/constructors.jl:47-57`, immediately before the closing `"""`:

```
Pass `names` (a vector of `Symbol`s) to label the elements of a vector random variable;
this is only valid when the result has `N == 1`.
```

Replace the two named overloads at `src/chains.jl:27-34` with:

```julia
function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{Symbol}) where {T}
    rd = from_chains(array)
    d = draws(rd)
    RandomDraw{T, 1, typeof(d)}(d, nchains(rd), param_names)
end

function from_chains(array::AbstractArray{T, 3}, param_names::AbstractVector{String}) where {T}
    from_chains(array, Symbol.(param_names))
end
```

Replace the `param_names` paragraph in the `from_chains` docstring at `src/chains.jl:10-12` with:

```
When `param_names` (a vector of `Symbol`s or `String`s) is given, the names are attached to
the returned value and are readable with [`variables`](@ref); `param_names` must have one
entry per variable.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. Both `Attaching names` and the updated `from_chains (MCMCChains interop)` report 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/constructors.jl src/chains.jl test/runtests.jl
git commit -m "Attach parameter names via RandomDraw kwarg and from_chains

from_chains(array, names) now returns a single named RandomDraw rather
than an (rd, names) tuple. Breaking, and the point of the change."
```

---

### Task 4: Name-based and names-preserving indexing

**Files:**
- Modify: `src/abstractarray.jl` (insert after the logical-indexing method at lines 22-28, before `setindex!`)
- Test: `test/runtests.jl` (insert before the `if HAS_MCMCCHAINS` guard)

**Interfaces:**
- Consumes: `_flat_store` (`src/abstractarray.jl:7`), the `names` field and `variables` from Task 2, `from_chains(array, names)` from Task 3.
- Produces: `Base.getindex` methods on `RandomDraw{T,1}` for `AbstractVector{<:Integer}`, `Colon`, `AbstractVector{Bool}`, `Symbol`, and `AbstractVector{Symbol}`; internal helper `_name_index(x, s) -> Int`.

**Critical dispatch note:** `Bool <: Integer` in Julia. Adding the `AbstractVector{<:Integer}` method **silently hijacks** logical indexing on a vector RV — verified: `x[[true, false, true]]` dispatches to the integer method rather than to the existing `getindex(::RandomDraw, ::AbstractArray{Bool})` at `src/abstractarray.jl:22`, and Julia raises no ambiguity error. The `AbstractVector{Bool}` method below is what preserves current behavior. Do not omit it.

- [ ] **Step 1: Write the failing test**

Insert into `test/runtests.jl` immediately before the `if HAS_MCMCCHAINS` line:

```julia
    @testset "Name-based and names-preserving indexing" begin
        d = reshape(collect(1.0:12.0), 4, 3)
        x = RandomDraw(d; names=[:a, :b, :c])

        # Indexing by name gives the scalar RV for that parameter.
        @test draws(x[:b]) == d[:, 2]
        @test x[:b] isa RandomDraw{Float64, 0}
        @test variables(x[:b]) === nothing        # a scalar RV has no elements to name

        # A vector of names selects and reorders, carrying the names along.
        sub = x[[:c, :a]]
        @test variables(sub) == [:c, :a]
        @test draws(sub) == d[:, [3, 1]]

        # Integer-vector and range indexing subset the names.
        @test variables(x[2:3]) == [:b, :c]
        @test draws(x[2:3]) == d[:, 2:3]
        @test variables(x[[1, 3]]) == [:a, :c]

        # x[:] keeps every name rather than silently dropping them.
        @test variables(x[:]) == [:a, :b, :c]
        @test draws(x[:]) == d

        # Logical indexing still works and also subsets names.
        @test variables(x[[true, false, true]]) == [:a, :c]
        @test draws(x[[true, false, true]]) == d[:, [1, 3]]

        # Unknown names error, and the message lists what is available.
        err = try; x[:zzz]; catch e; sprint(showerror, e); end
        @test occursin("zzz", err)
        @test occursin("a", err) && occursin("b", err) && occursin("c", err)

        # Name indexing on an unnamed RV errors rather than returning nonsense.
        u = RandomDraw(d)
        @test_throws ErrorException u[:a]

        # Subsetting an unnamed RV still works and stays unnamed.
        @test variables(u[2:3]) === nothing
        @test draws(u[2:3]) == d[:, 2:3]

        # nchains survives subsetting.
        xc = RandomDraw(d; nchains=2, names=[:a, :b, :c])
        @test nchains(xc[2:3]) == 2
        @test nchains(xc[:b]) == 2
    end

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL. `Name-based and names-preserving indexing` errors on `x[:b]` with `MethodError: no method matching getindex(::RandomDraw{Float64, 1, Matrix{Float64}}, ::Symbol)`.

- [ ] **Step 3: Write the implementation**

Insert into `src/abstractarray.jl` after the logical-indexing method (lines 22-28), before the first `setindex!`:

```julia
# Subsetting a vector RV. The generic AbstractArray fallback would route through
# `similar`, which receives only a type and a shape and so cannot carry the names;
# slicing the flat store directly both preserves them and skips the fallback's
# element-by-element loop.
function Base.getindex(x::RandomDraw{T, 1}, I::AbstractVector{<:Integer}) where {T}
    @boundscheck checkbounds(x, I)
    data = _flat_store(x)[:, I]
    nms = x.names === nothing ? nothing : x.names[I]
    RandomDraw{T, 1, typeof(data)}(data, x.nchains, nms)
end

Base.getindex(x::RandomDraw{T, 1}, ::Colon) where {T} = x[1:length(x)]

# Bool <: Integer, so without this method a logical index would silently dispatch to the
# integer method above and be read as positions 1 and 0. Julia reports no ambiguity here.
function Base.getindex(x::RandomDraw{T, 1}, idx::AbstractVector{Bool}) where {T}
    length(idx) == length(x) ||
        throw(DimensionMismatch("logical index length $(length(idx)) != length $(length(x))"))
    return x[findall(idx)]
end

function _name_index(x::RandomDraw, s::Symbol)
    nms = x.names
    nms === nothing && error("This RandomDraw has no parameter names")
    i = findfirst(isequal(s), nms)
    i === nothing &&
        error("Unknown parameter name :$s; available: $(join(string.(nms), \", \"))")
    return i
end

Base.getindex(x::RandomDraw{T, 1}, s::Symbol) where {T} = x[_name_index(x, s)]

function Base.getindex(x::RandomDraw{T, 1}, S::AbstractVector{Symbol}) where {T}
    return x[[_name_index(x, s) for s in S]]
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. `Name-based and names-preserving indexing` reports 0 failures, and the pre-existing `Indexing value correctness (C2/H1/H2)` testset still passes — confirming the new `AbstractVector{Bool}` method preserved logical indexing.

- [ ] **Step 5: Commit**

```bash
git add src/abstractarray.jl test/runtests.jl
git commit -m "Add name-based indexing and preserve names through subsetting"
```

---

### Task 5: Propagate names through arithmetic and broadcasting

**Files:**
- Modify: `src/types.jl` (append after `_combine_nchains`, which ends at line 114)
- Modify: `src/arithmetic.jl:1-11` (`_binop_scalar`), `:13-31` (`_binop_rv`), `:89-90` (unary ops), `:92-99` (the `@eval` loop)
- Modify: `src/broadcast.jl:26-49` (`similar(::Broadcasted, ::Type)`)
- Modify: `src/abstractarray.jl` (the `Base.copy` method, currently at lines 62-64)
- Test: `test/runtests.jl` (insert before the `if HAS_MCMCCHAINS` guard)

**Interfaces:**
- Consumes: the `names` field from Task 2; `RandomDraw(...; names=...)` from Task 3.
- Produces: `_combine_names(operands...) -> Union{Nothing, Vector{Symbol}}` and `_maybe_names(x::RandomDraw, nms) -> RandomDraw`, both internal to `src/types.jl`.

- [ ] **Step 1: Write the failing test**

Insert into `test/runtests.jl` immediately before the `if HAS_MCMCCHAINS` line:

```julia
    @testset "Name propagation" begin
        d = reshape(collect(1.0:12.0), 4, 3)
        x = RandomDraw(d; names=[:a, :b, :c])
        y = RandomDraw(d .* 2; names=[:a, :b, :c])
        z = RandomDraw(d .* 3; names=[:p, :q, :r])
        u = RandomDraw(d .* 4)                       # multi-draw, unnamed
        k = as_rs([10.0, 20.0, 30.0])                # single-draw constant, unnamed

        # Preserved: elementwise ops against a scalar.
        @test variables(x .+ 1) == [:a, :b, :c]
        @test variables(x .* 2) == [:a, :b, :c]
        @test variables(2 .- x) == [:a, :b, :c]
        @test variables(sin(x)) == [:a, :b, :c]
        @test variables(-x) == [:a, :b, :c]
        @test variables(sin.(x)) == [:a, :b, :c]
        @test variables(x .* 2 .+ 1) == [:a, :b, :c]   # fused broadcast

        # Preserved: two operands whose names agree.
        @test variables(x .+ y) == [:a, :b, :c]
        @test variables(x + y) == [:a, :b, :c]

        # Preserved: the other operand is a single-draw constant, so name-agnostic.
        @test variables(x .+ k) == [:a, :b, :c]

        # Dropped: names disagree, or the other operand is multi-draw and unnamed.
        @test variables(x .+ z) === nothing
        @test variables(x .+ u) === nothing

        # Values must be untouched by any of this.
        @test draws(x .+ y) ≈ d .+ (d .* 2)
        @test draws(x .+ z) ≈ d .+ (d .* 3)

        # copy preserves; similar and reshape drop.
        @test variables(copy(x)) == [:a, :b, :c]
        @test variables(similar(x)) === nothing
        @test variables(reshape(x, (3,))) === nothing

        # Shape-collapsing operations drop names.
        @test variables(rs_mean(x)) === nothing
        A = RandomDraw(randn(4, 2, 3))
        @test variables(A * x) === nothing

        # nchains bookkeeping is unaffected by the names plumbing.
        xc = RandomDraw(d; nchains=2, names=[:a, :b, :c])
        @test nchains(xc .+ 1) == 2
        @test variables(xc .+ 1) == [:a, :b, :c]
    end

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL. `Name propagation` fails at the first assertion, `variables(x .+ 1) == [:a, :b, :c]`, because broadcasting reconstructs without names and `variables` returns `nothing`.

- [ ] **Step 3: Write the implementation**

Append to `src/types.jl`, after `_combine_nchains`:

```julia
# Combine parameter names across operands of an elementwise/broadcast operation, with the
# same rule as _combine_nchains: a single-draw operand is a constant with no identity of
# its own and defers; two genuine but differing name sets cannot be reconciled, so the
# result is unnamed.
function _combine_names(operands...)
    nms = nothing
    seen = false
    for x in operands
        x isa RandomDraw || continue
        size(x.draws, 1) == 1 && continue  # constant: name-agnostic
        if !seen
            nms = x.names
            seen = true
        elseif !isequal(nms, x.names)
            nms = nothing
        end
    end
    return nms
end

# Attach `nms` to `x` only where it still describes the result elementwise. Broadcasting
# can change both rank and length (a named length-3 vector RV times a 3x2 matrix RV gives
# an N=2 result), and stale names are worse than none.
function _maybe_names(x::RandomDraw{T, N}, nms) where {T, N}
    (nms === nothing || N != 1 || length(nms) != length(x)) && return x
    return RandomDraw{T, N, typeof(x.draws)}(x.draws, x.nchains, nms)
end
```

Replace `src/arithmetic.jl:1-11` (both `_binop_scalar` methods) with:

```julia
function _binop_scalar(f::Function, x::RandomDraw, y::Number)
    d = draws(x)
    result = f.(d, y)
    _maybe_names(RandomDraw(result, nchains=nchains(x)), x.names)
end

function _binop_scalar(f::Function, x::Number, y::RandomDraw)
    d = draws(y)
    result = f.(x, d)
    _maybe_names(RandomDraw(result, nchains=nchains(y)), y.names)
end
```

Replace the final two lines of `_binop_rv` (`src/arithmetic.jl:29-30`):

```julia
    result = f.(dx, dy)
    RandomDraw(result, nchains=_combine_nchains(x, y))
```

with:

```julia
    result = f.(dx, dy)
    _maybe_names(RandomDraw(result, nchains=_combine_nchains(x, y)), _combine_names(x, y))
```

Replace the unary operators at `src/arithmetic.jl:89-90` with:

```julia
Base.:!(x::RandomDraw) = _maybe_names(RandomDraw(.!(draws(x)), nchains=nchains(x)), x.names)
Base.:-(x::RandomDraw) = _maybe_names(RandomDraw(-(draws(x)), nchains=nchains(x)), x.names)
```

Replace the body of the `@eval` loop at `src/arithmetic.jl:96-98` with:

```julia
    @eval begin
        Base.$f(x::RandomDraw) =
            _maybe_names(RandomDraw($f.(draws(x)), nchains=nchains(x)), x.names)
    end
```

Replace the final three lines of `similar` in `src/broadcast.jl:45-48`:

```julia
    nc = _combine_nchains(bcf.args...)
    sz = (n_draws, length.(axs)...)
    data = similar(src_draws, T, sz)
    RandomDraw{T, N, typeof(data)}(data, nc)
```

with:

```julia
    nc = _combine_nchains(bcf.args...)
    sz = (n_draws, length.(axs)...)
    data = similar(src_draws, T, sz)
    return _maybe_names(RandomDraw{T, N, typeof(data)}(data, nc), _combine_names(bcf.args...))
```

Replace `Base.copy` at `src/abstractarray.jl:62-64` with:

```julia
function Base.copy(x::RandomDraw{T, N}) where {T, N}
    RandomDraw{T, N, typeof(x.draws)}(copy(x.draws), x.nchains, x.names)
end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. `Name propagation` reports 0 failures. Pay particular attention to the pre-existing `Fused broadcast expressions`, `Broadcasting over N>=2 RVs (H3)` and `nchains propagation (H5/H6/M6)` testsets — they exercise the same code paths and must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/arithmetic.jl src/broadcast.jl src/abstractarray.jl test/runtests.jl
git commit -m "Propagate parameter names through elementwise ops and broadcasting"
```

---

### Task 6: Label `show` output with names

**Files:**
- Modify: `src/abstractarray.jl` (the `N == 1` branch of `Base.show`, currently at lines 82-86)
- Test: `test/runtests.jl` (insert before the `if HAS_MCMCCHAINS` guard)

**Interfaces:**
- Consumes: the `names` field from Task 2 and `RandomDraw(...; names=...)` from Task 3.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing test**

Insert into `test/runtests.jl` immediately before the `if HAS_MCMCCHAINS` line:

```julia
    @testset "show with names" begin
        d = reshape(collect(1.0:12.0), 4, 3)
        named = RandomDraw(d; names=[:alpha, :beta, :gamma])
        s = sprint(show, named)
        @test occursin("[alpha]", s)
        @test occursin("[beta]", s)
        @test occursin("[gamma]", s)

        # Unnamed values keep the positional labels.
        plain = sprint(show, RandomDraw(d))
        @test occursin("[1]", plain)
        @test !occursin("[alpha]", plain)
    end

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL. `show with names` fails on `occursin("[alpha]", s)` — `show` still prints positional labels `[1]`, `[2]`, `[3]`.

- [ ] **Step 3: Write the implementation**

Replace the `N == 1` branch of `Base.show` at `src/abstractarray.jl:82-86`:

```julia
        if N == 1
            for i in 1:n_show
                print(io, "\n[$i] ", round(m_vals[i]; digits=2), " ± ", round(s_vals[i]; digits=2))
            end
```

with:

```julia
        if N == 1
            nms = x.names
            for i in 1:n_show
                lbl = nms === nothing ? string(i) : string(nms[i])
                print(io, "\n[", lbl, "] ",
                      round(m_vals[i]; digits=2), " ± ", round(s_vals[i]; digits=2))
            end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. `show with names` reports 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/abstractarray.jl test/runtests.jl
git commit -m "Label show output with parameter names when present"
```

---

### Task 7: Attach names automatically in the MCMCChains extension

**Files:**
- Modify: `ext/RandomDrawsMCMCChainsExt/RandomDrawsMCMCChainsExt.jl`
- Test: `test/runtests.jl` (append inside the existing `@testset "MCMCChains extension value correctness (H7)"`, which sits inside the `if HAS_MCMCCHAINS` guard)

**Interfaces:**
- Consumes: `from_chains(array, param_names)` from Task 3, `variables` from Task 2.
- Produces: nothing other tasks depend on. This is the final task.

**Note on running these tests:** under `julia --project=. test/runtests.jl`, `HAS_MCMCCHAINS` is `false` and this testset skips, so that command cannot verify this task. Use the temporary-environment script in Step 2 instead. `Pkg.test()` does not work on this machine (compressed General registry).

- [ ] **Step 1: Write the failing test**

Append inside the existing `@testset "MCMCChains extension value correctness (H7)"` block in `test/runtests.jl`, after the final `@test draws(from_chains(chn)) == draws(rd)` assertion:

```julia
            # Parameter names come straight off the Chains object.
            @test variables(rd) == [:a, :b, :cc]
            @test variables(from_chains(chn)) == [:a, :b, :cc]

            # Names line up with the right columns, so name lookup and position agree.
            for (j, nm) in enumerate([:a, :b, :cc])
                @test draws(rd[nm]) == draws(rd)[:, j]
            end
```

- [ ] **Step 2: Run the test to verify it fails**

Create the runner script (this environment is where the extension actually loads):

```bash
mkdir -p "$CLAUDE_JOB_DIR/tmp"
cat > "$CLAUDE_JOB_DIR/tmp/runext.jl" <<'EOF'
import Pkg
Pkg.activate(; temp=true, io=devnull)
Pkg.develop(path="/media/karim/Code-Drive/karimn-code/RandomDraws.jl", io=devnull)
Pkg.add("MCMCChains", io=devnull)
include("/media/karim/Code-Drive/karimn-code/RandomDraws.jl/test/runtests.jl")
EOF
```

Run: `julia --startup-file=no "$CLAUDE_JOB_DIR/tmp/runext.jl"`

Expected: FAIL. `MCMCChains extension value correctness (H7)` fails on `variables(rd) == [:a, :b, :cc]` — the extension calls the single-argument `from_chains`, so `variables(rd)` is `nothing`.

- [ ] **Step 3: Write the implementation**

Replace the whole of `ext/RandomDrawsMCMCChainsExt/RandomDrawsMCMCChainsExt.jl` with:

```julia
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --startup-file=no "$CLAUDE_JOB_DIR/tmp/runext.jl"`

Expected: PASS. Every testset reports 0 failures and 0 errors, including all the extension testsets.

Then confirm the standalone path is still green:

Run: `julia --project=. test/runtests.jl`

Expected: PASS, with the `@info "MCMCChains not available; skipping extension tests (run via Pkg.test)"` message.

- [ ] **Step 5: Commit**

```bash
git add ext/RandomDrawsMCMCChainsExt/RandomDrawsMCMCChainsExt.jl test/runtests.jl
git commit -m "Attach Chains parameter names automatically in the MCMCChains extension"
```

---

## Self-Review Notes

Spec coverage check — every spec requirement maps to a task:

| Spec requirement | Task |
|---|---|
| `collect` returns `Array{RandomDraw{T,0},N}` | 1 |
| `Array` matches `collect` | 1 |
| `map` routes to broadcast, multi-arg checks shapes | 1 |
| `isequal` returns `Bool` | 2 |
| Docstring "Deviations" section | 2 |
| `names` field, optional third constructor argument | 2 |
| `N == 1` and length invariants | 2 |
| `variables` accessor, exported | 2 |
| `from_chains(array, names)` returns one value | 3 |
| `RandomDraw(...; names=)` kwarg | 3 |
| MCMCChains attaches names | 7 |
| Propagation table (preserve/drop/subset) | 4 (subset), 5 (preserve/drop) |
| `_combine_names` mirroring `_combine_nchains` | 5 |
| Name-based indexing, unknown-name error | 4 |
| Integer-vector and `Colon` methods | 4 |
| `show` labels with names | 6 |
| Four test areas from the spec's Testing section | 1, 4, 5, 7 (plus 2, 3, 6 added during planning) |

One addition beyond the spec, made during planning: Task 4's `AbstractVector{Bool}` method. It is not in the spec but is **required for correctness** — `Bool <: Integer` means the new integer-vector method would otherwise silently capture logical indexing and misread `[true, false, true]` as positions. This was verified experimentally, not assumed.

**Correction, made during Task 2 execution.** An earlier draft of this plan recorded "no custom `Base.hash`" as a deliberate omission, on the claim that Base's content-based `hash(::AbstractArray)` was already adequate and had been checked not to recurse. That check was invalid — it ran `hash` inside an `@async` task and tested `istaskdone`, which returns `true` for a failed task as well as a completed one, so it reported success on a stack overflow.

`hash` in fact throws `StackOverflowError` on **any** `RandomDraw`: Base's generic method hashes elements, an element of a `RandomDraw` is another `RandomDraw`, and for `N == 0` that is itself. Task 2 therefore also adds, in `src/abstractarray.jl` immediately after `isequal`:

```julia
Base.hash(x::RandomDraw, h::UInt) = hash(x.names, hash(x.nchains, hash(x.draws, hash(:RandomDraw, h))))
```

It hashes exactly the three fields `isequal` compares, so `isequal(x, y)` implies `hash(x) == hash(y)`. This is a scope addition beyond the original Task 2 brief, fixing a pre-existing latent bug that adding `isequal` surfaced.
