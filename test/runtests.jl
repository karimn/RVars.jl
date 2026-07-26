using RVars
using Statistics
using LinearAlgebra
using Random
using Test

# MCMCChains is a test-target dependency that exercises the package extension.
# Guard it so the suite still runs under a plain `julia --project=. test/runtests.jl`
# (where MCMCChains is not in the manifest); Pkg.test provides it via [targets].
const HAS_MCMCCHAINS = try
    @eval import MCMCChains
    true
catch
    false
end

# Tables is a test-only dependency, used solely for a Tables.jl-interface smoke test on
# gather_draws; guard it the same way as MCMCChains above.
const HAS_TABLES = try
    @eval import Tables
    true
catch
    false
end

@testset "RVars.jl" begin

    @testset "Constructors" begin
        x = RVar(randn(1000))
        @test x isa RVar{Float64, 0}
        @test ndraws(x) == 1000
        @test nchains(x) == 1
        @test length(x) == 1

        y = RVar(randn(1000, 3))
        @test y isa RVar{Float64, 1}
        @test size(y) == (3,)
        @test length(y) == 3

        z = RVar(randn(1000, 4, 3))
        @test z isa RVar{Float64, 2}
        @test size(z) == (4, 3)

        w = RVar(randn(800, 3), nchains=4)
        @test nchains(w) == 4
        @test niterations(w) == 200
        @test ndraws(w) == 800
    end

    @testset "with_chains" begin
        data = randn(200, 3)
        x = RVar(data; with_chains=true)
        @test nchains(x) == 3
        @test niterations(x) == 200
        @test ndraws(x) == 600
        @test size(x) == ()

        # An explicit nchains conflicts with with_chains and is ignored with a warning.
        x2 = @test_logs (:warn,) RVar(data; with_chains=true, nchains=5)
        @test nchains(x2) == 3
        # No warning when nchains is left at its default.
        @test_logs RVar(data; with_chains=true)
    end

    @testset "as_rs" begin
        x = as_rs([1.0, 2.0, 3.0])
        @test x isa RVar{Float64, 1}
        @test ndraws(x) == 1
        @test size(x) == (3,)

        s = as_rs(5.0)
        @test s isa RVar{Float64, 0}
        @test ndraws(s) == 1

        m = as_rs(ones(2, 3))
        @test m isa RVar{Float64, 2}
        @test size(m) == (2, 3)
    end

    @testset "AbstractArray" begin
        x = RVar(randn(1000, 3))
        @test size(x) == (3,)

        el = x[1]
        @test el isa RVar{Float64, 0}
        @test ndraws(el) == 1000

        @test length(x) == 3
        @test axes(x) == (Base.OneTo(3),)
        @test ndims(x) == 1
    end

    @testset "Arithmetic" begin
        x = RVar(rand(1000))
        y = RVar(rand(1000))
        @test x + y isa RVar
        @test x - y isa RVar
        @test x * y isa RVar
        @test x / y isa RVar
        @test x + 1.0 isa RVar
        @test 1.0 + x isa RVar
        @test x - 1.0 isa RVar
        @test 1.0 - x isa RVar
        @test x * 2.0 isa RVar
        @test 2.0 * x isa RVar
        @test x / 2.0 isa RVar
        @test 2.0 \ x isa RVar
        @test -x isa RVar
        @test x ^ 2 isa RVar
        for f in [sin, cos, tan, exp, log, abs, sqrt, floor, ceil, round, sign]
            @test f(x) isa RVar
        end
        @test x .+ y isa RVar
        x_vec = RVar(rand(1000, 3))
        @test (x_vec .> 0.5) isa RVar{Bool, 1}
    end

    @testset "Stats over draws" begin
        x = RVar(randn(1000, 3))
        @test mean(x) isa Vector{Float64}
        @test length(mean(x)) == 3
        @test std(x) isa Vector{Float64}
        @test var(x) isa Vector{Float64}
        @test median(x) isa Vector{Float64}

        x_scalar = RVar(randn(1000))
        @test mean(x_scalar) isa Float64
        @test std(x_scalar) isa Float64
        @test var(x_scalar) isa Float64
        @test median(x_scalar) isa Float64
    end

    @testset "E and Pr" begin
        x = RVar(randn(10000, 3))
        @test E(x) ≈ mean(x)
        gt = x .> 0
        @test gt isa RVar{Bool, 1}
        prob = mean(gt)
        @test prob isa Vector{Float64}
        @test all(0.4 .< prob .< 0.6)
    end

    @testset "rs_ summaries" begin
        x = RVar(randn(1000, 4, 3))
        @test rs_mean(x) isa RVar{Float64, 0}
        @test ndraws(rs_mean(x)) == 1000
        @test rs_sum(x) isa RVar
        @test rs_sd(x) isa RVar
        @test rs_var(x) isa RVar
        @test rs_median(x) isa RVar
        @test rs_min(x) isa RVar
        @test rs_max(x) isa RVar

        y = RVar(randn(1000))
        @test rs_quantile(y, [0.25, 0.5, 0.75]) isa RVar{Float64, 1}
    end

    @testset "Constants" begin
        y = RVar(randn(1000))
        z = as_rs(1.0) + y
        @test ndraws(z) == 1000

        c = as_rs(ones(3))
        z2 = c + RVar(randn(1000, 3))
        @test ndraws(z2) == 1000
        @test size(z2) == (3,)

        z3 = y + 1.0
        @test ndraws(z3) == 1000
        m = mean(z3)
        @test m isa Float64
    end

    @testset "rvar_rng" begin
        x = rvar_rng(randn, 3)
        @test x isa RVar{Float64, 1}
        @test size(x) == (3,)
        @test ndraws(x) == 2000
    end

    @testset "from_chains (MCMCChains interop)" begin
        data = randn(200, 5, 4)
        rd = from_chains(data)
        @test rd isa RVar{Float64, 1}
        @test size(rd) == (5,)
        @test ndraws(rd) == 800
        @test nchains(rd) == 4
        @test niterations(rd) == 200

        # With names, the default is one random variable per model parameter.
        rd2 = from_chains(data, [:mu, :sigma, :alpha, :beta, :lp])
        @test rd2 isa NamedTuple
        @test keys(rd2) == (:mu, :sigma, :alpha, :beta, :lp)
        @test all(v -> v isa RVar{Float64, 0}, values(rd2))

        rd3 = from_chains(data, ["mu", "sigma", "alpha", "beta", "lp"])
        @test keys(rd3) == (:mu, :sigma, :alpha, :beta, :lp)

        # flat=true keeps the ungrouped named vector RVar.
        rd4 = from_chains(data, [:mu, :sigma, :alpha, :beta, :lp]; flat=true)
        @test rd4 isa RVar{Float64, 1}
        @test variables(rd4) == [:mu, :sigma, :alpha, :beta, :lp]
    end

    @testset "Matrix multiplication" begin
        A = RVar(randn(1000, 2, 3))
        B = RVar(randn(1000, 3, 4))
        C = A * B
        @test C isa RVar{Float64, 2}
        @test size(C) == (2, 4)
        @test ndraws(C) == 1000

        v = RVar(randn(1000, 3))
        w = RVar(randn(1000, 3))
        d = dot(v, w)
        @test d isa RVar{Float64, 0}

        M = RVar(randn(1000, 3, 2))
        v2 = RVar(randn(1000, 2))
        mv = M * v2
        @test mv isa RVar{Float64, 1}
        @test size(mv) == (3,)
    end

    @testset "Linear algebra and quantile fixes (H8/M2/M3)" begin
        nd = 20

        # H8: vector-RV × matrix-RV is a per-draw row-vector × matrix (length m == k).
        vx = randn(nd, 3)
        my = randn(nd, 3, 2)
        vm = RVar(vx) * RVar(my)
        @test vm isa RVar{Float64, 1}
        @test size(vm) == (2,)
        @test ndraws(vm) == nd
        for i in 1:nd
            @test draws(vm)[i, :] ≈ vec(vx[i, :]' * my[i, :, :])
        end

        # M2: rs_quantile reduces per draw over elements, like the rest of the rs_ family.
        xm = RVar(randn(nd, 5))
        q = rs_quantile(xm, [0.0, 0.5, 1.0])
        @test q isa RVar{Float64, 1}
        @test size(q) == (3,)
        @test ndraws(q) == nd
        dm = draws(xm)
        for i in 1:nd
            @test draws(q)[i, :] ≈ quantile(dm[i, :], [0.0, 0.5, 1.0])
        end

        # M3: summarise-over-draws quantile with a vector of probabilities.
        xv = RVar(randn(nd, 4))
        qs = quantile(xv, [0.25, 0.5, 0.75])
        @test qs isa AbstractMatrix
        @test size(qs) == (3, 4)   # (probabilities, elements)
        dv = draws(xv)
        for e in 1:4
            @test qs[:, e] ≈ quantile(dv[:, e], [0.25, 0.5, 0.75])
        end
        # scalar probability still reduces to a per-element vector.
        @test quantile(xv, 0.5) isa AbstractVector
        @test length(quantile(xv, 0.5)) == 4
    end

    @testset "Mixed RVar / plain-array linear algebra" begin
        nd = 15
        dv = randn(nd, 3)        # vector RV of length 3
        dM = randn(nd, 2, 3)     # matrix RV, 2x3
        v = RVar(dv)
        Mv = RVar(dM)
        A = randn(3, 4)
        B = randn(5, 3)
        pv = randn(3)

        # vector RV × plain matrix -> length 4
        r1 = v * A
        @test r1 isa RVar{Float64, 1}
        @test size(r1) == (4,)
        for i in 1:nd
            @test draws(r1)[i, :] ≈ vec(dv[i, :]' * A)
        end

        # plain matrix × vector RV -> length 5
        r2 = B * v
        @test r2 isa RVar{Float64, 1}
        @test size(r2) == (5,)
        for i in 1:nd
            @test draws(r2)[i, :] ≈ B * dv[i, :]
        end

        # matrix RV × plain vector -> length 2
        r3 = Mv * pv
        @test r3 isa RVar{Float64, 1}
        @test size(r3) == (2,)
        for i in 1:nd
            @test draws(r3)[i, :] ≈ dM[i, :, :] * pv
        end

        # dot against a plain vector, both orders
        r4 = dot(v, pv)
        @test r4 isa RVar{Float64, 0}
        for i in 1:nd
            @test draws(r4)[i] ≈ dot(dv[i, :], pv)
        end
        r5 = dot(pv, v)
        @test r5 isa RVar{Float64, 0}
        for i in 1:nd
            @test draws(r5)[i] ≈ dot(pv, dv[i, :])
        end
    end

    @testset "Type stability" begin
        x = RVar(rand(1000))
        y = x + x
        @test y isa RVar{Float64}
    end

    @testset "eltype preservation (M4/M5/M7)" begin
        # M5: as_rs(::Number) preserves the number's type instead of forcing Float64.
        @test as_rs(3) isa RVar{Int, 0}
        @test as_rs(0.1f0) isa RVar{Float32, 0}
        @test as_rs(1 + 2im) isa RVar{Complex{Int}, 0}
        @test draws(as_rs(3)) == [3]
        @test as_rs(2.0) isa RVar{Float64, 0}   # still works

        # M4: rvar_rng preserves the sampler's eltype.
        r = rvar_rng(n -> rand(1:6, n), 3; ndraws=500)
        @test r isa RVar{Int, 1}
        @test size(r) == (3,)
        @test ndraws(r) == 500

        # M7: rand honors and validates the requested dimensionality N.
        rng = MersenneTwister(0)
        a = rand(rng, RVar{Float64, 2}, (2, 3), 100)
        @test a isa RVar{Float64, 2}
        @test size(a) == (2, 3)
        @test ndraws(a) == 100
        @test_throws ErrorException rand(rng, RVar{Float64, 2}, (3,), 100)
    end

    @testset "nchains propagation (H5/H6/M6)" begin
        x = RVar(randn(800, 3), nchains=4)
        @test nchains(x) == 4

        # H5: combining with a constant (1-draw) operand must not collapse nchains.
        c = as_rs(2.0)
        @test nchains(x + c) == 4
        @test nchains(c + x) == 4
        @test nchains(x .+ c) == 4
        @test nchains(x * 2.0) == 4
        @test nchains(2.0 + x) == 4
        @test nchains(sin(x)) == 4

        # Two genuinely different chain counts cannot align -> collapse to 1.
        y = RVar(randn(800, 3), nchains=2)
        @test nchains(x + y) == 1

        # H6: nchains kwarg must be honored for flat-vector input.
        v = RVar(collect(1.0:800.0); nchains=4)
        @test nchains(v) == 4
        @test niterations(v) == 200

        # M6: a single-draw object cannot claim multiple chains.
        @test_throws ErrorException RVar(randn(1, 3); nchains=2)
    end

    @testset "Indexing value correctness (C2/H1/H2)" begin
        nd = 5
        # store[k, i, j] distinguishable per (draw, row, col); visible shape (2, 3).
        store = [1000k + 10i + j for k in 1:nd, i in 1:2, j in 1:3]
        x = RVar(store)
        @test x isa RVar{Int, 2}
        @test size(x) == (2, 3)
        @test length(x) == 6

        # Cartesian access returns the correct element's draws (H1: column-major).
        for i in 1:2, j in 1:3
            el = x[i, j]
            @test el isa RVar{Int, 0}
            @test draws(el) == store[:, i, j]
        end

        # Linear indexing is column-major and agrees with Cartesian.
        li = LinearIndices((2, 3))
        for i in 1:2, j in 1:3
            @test draws(x[li[i, j]]) == draws(x[i, j])
        end
        @test draws(x[3]) == store[:, 1, 2]   # column-major: 3 -> (1,2)

        # H2: out-of-range indices must throw, not clamp to the last element.
        @test_throws BoundsError x[7]
        @test_throws BoundsError x[0]
        v = RVar(randn(50, 3))          # N=1
        @test_throws BoundsError v[4]
        @test_throws BoundsError v[100]

        # Materializing elements into scalar RVs via a comprehension round-trips.
        c = [x[i, j] for i in 1:2, j in 1:3]
        @test c isa Matrix{<:RVar}
        @test draws(c[2, 3]) == store[:, 2, 3]

        # Logical indexing selects the right columns (flattened, column-major).
        mask = [true false true; false true false]   # shape (2,3)
        sel = x[mask]
        @test sel isa RVar{Int, 1}
        @test size(sel) == (3,)
        expected_cols = findall(vec(mask))
        flat = reshape(store, nd, 6)
        @test draws(sel) == flat[:, expected_cols]

        # setindex! (scalar) writes across all draws of one element.
        y = RVar(zeros(nd, 2, 3))
        y[li[2, 3]] = 7.0
        @test all(draws(y[2, 3]) .== 7.0)
        @test all(draws(y[1, 1]) .== 0.0)
        @test_throws BoundsError (y[7] = 1.0)
    end

    @testset "Broadcasting over N>=2 RVs (H3)" begin
        nd = 8
        store = reshape(collect(1.0:(nd * 6)), nd, 2, 3)
        x = RVar(store)

        s = sin.(x)
        @test s isa RVar{Float64, 2}
        @test size(s) == (2, 3)
        @test draws(s[1, 2]) == sin.(store[:, 1, 2])

        a = x .+ 0.0
        @test draws(a[2, 3]) == store[:, 2, 3]

        t = x .+ x
        @test draws(t[2, 1]) == store[:, 2, 1] .+ store[:, 2, 1]

        # Broadcasting a scalar-RV against a vector-RV expands the singleton.
        c = as_rs(10.0)                        # N=0, 1 draw
        w = RVar(randn(nd, 3))
        cw = c .+ w
        @test cw isa RVar{Float64, 1}
        @test size(cw) == (3,)
    end

    @testset "Fused broadcast expressions" begin
        nd = 40
        d = randn(nd, 3)
        x = RVar(d)

        r1 = sin.(x) .+ 1.0
        @test r1 isa RVar{Float64, 1}
        @test size(r1) == (3,)
        @test draws(r1[2]) ≈ sin.(d[:, 2]) .+ 1.0

        r2 = x .* 2 .+ 1
        @test r2 isa RVar{Float64, 1}
        @test draws(r2[1]) ≈ d[:, 1] .* 2 .+ 1

        r3 = 1.0 .+ (x .* 2)
        @test r3 isa RVar{Float64, 1}
        @test draws(r3[3]) ≈ 1.0 .+ (d[:, 3] .* 2)

        # Fused expression combining two random variables.
        y = RVar(randn(nd, 3))
        dy = draws(y)
        r4 = (x .+ y) .* 2
        @test r4 isa RVar{Float64, 1}
        @test draws(r4[1]) ≈ (d[:, 1] .+ dy[:, 1]) .* 2

        # Fused expression over an N>=2 RV.
        m = RVar(randn(nd, 2, 3))
        dm = draws(m)
        r5 = m .* 2 .+ 1
        @test r5 isa RVar{Float64, 2}
        @test size(r5) == (2, 3)
        @test draws(r5[2, 3]) ≈ dm[:, 2, 3] .* 2 .+ 1

        # nchains survives a fused expression.
        xc = RVar(randn(800, 3), nchains=4)
        @test nchains(xc .* 2 .+ 1) == 4
    end

    @testset "from_chains value correctness (C1)" begin
        # A[i, v, c] carries a distinguishable value per (iteration, variable, chain).
        n_iter, n_var, n_chain = 2, 3, 4
        A = [100c + 10v + i for i in 1:n_iter, v in 1:n_var, c in 1:n_chain]
        rd = from_chains(A)
        @test nchains(rd) == n_chain
        @test niterations(rd) == n_iter
        @test size(rd) == (n_var,)

        # draws(; with_chains) must reshape the flat draw axis back to (iter, chain, shape...)
        # with each sample landing exactly where it came from.
        wc = draws(rd; with_chains=true)
        @test size(wc) == (n_iter, n_chain, n_var)
        for i in 1:n_iter, v in 1:n_var, c in 1:n_chain
            @test wc[i, c, v] == A[i, v, c]
        end

        # A variable's flat column must gather that variable's samples only,
        # ordered iteration-fastest then chain.
        d = draws(rd)
        for v in 1:n_var
            expected = vec([A[i, v, c] for i in 1:n_iter, c in 1:n_chain])
            @test d[:, v] == expected
        end
    end

    @testset "AbstractArray contract holes (collect/Array/map)" begin
        d = reshape(collect(1.0:12.0), 4, 3)   # 4 draws, length-3 vector RV
        x = RVar(d)

        c = collect(x)
        @test c isa Vector
        @test size(c) == (3,)
        @test all(e -> e isa RVar, c)
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
        y = RVar(d2)
        c2 = collect(y)
        @test size(c2) == (2, 3)
        for i in 1:2, j in 1:3
            @test draws(c2[i, j]) == d2[:, i, j]
        end

        # map and broadcast must not disagree.
        m = map(sin, x)
        @test m isa RVar
        @test draws(m) ≈ sin.(d)
        @test draws(m) ≈ draws(sin.(x))

        # Multi-argument map.
        x2 = RVar(d .* 2)
        m2 = map(+, x, x2)
        @test m2 isa RVar
        @test draws(m2) ≈ d .+ (d .* 2)

        # map requires equal sizes; it must not silently broadcast-expand.
        z = RVar(reshape(collect(1.0:8.0), 4, 2))
        @test_throws DimensionMismatch map(+, x, z)
    end

    @testset "Names field, variables accessor, isequal" begin
        d = reshape(collect(1.0:12.0), 4, 3)

        # Default: no names, and existing 2-argument construction still works.
        x = RVar(d)
        @test variables(x) === nothing
        @test RVar{Float64, 1, typeof(d)}(d, 1) isa RVar

        # Names attach via the inner constructor's third positional argument.
        named = RVar{Float64, 1, typeof(d)}(d, 1, [:a, :b, :c])
        @test variables(named) == [:a, :b, :c]
        @test draws(named) == d
        @test nchains(named) == 1

        # Wrong number of names is rejected.
        @test_throws ErrorException RVar{Float64, 1, typeof(d)}(d, 1, [:a, :b])

        # Names are only valid for N == 1.
        d0 = collect(1.0:4.0)
        @test_throws ErrorException RVar{Float64, 0, typeof(d0)}(d0, 1, [:a])
        d2 = reshape(collect(1.0:24.0), 4, 2, 3)
        @test_throws ErrorException RVar{Float64, 2, typeof(d2)}(d2, 1, [:a, :b])

        # isequal returns a genuine Bool, unlike == which is elementwise by design.
        @test isequal(x, RVar(d)) === true
        @test (x == RVar(d)) isa RVar
        @test isequal(x, named) === false                       # names differ
        @test isequal(x, RVar(d, nchains=2)) === false     # nchains differ
        @test isequal(x, RVar(d .+ 1)) === false           # draws differ

        # Base's AbstractArray hash recurses forever on a scalar RV (x[1] of an N=0
        # value is another N=0 value), so RVar needs its own method.
        @test hash(x[1]) isa UInt
        @test hash(x) == hash(RVar(d))
        @test isequal(x, RVar(d)) && hash(x) == hash(RVar(d))

        # isequal makes RVar usable as a Dict key.
        dict = Dict(x => "unnamed", named => "named")
        @test dict[RVar(d)] == "unnamed"
        @test length(dict) == 2

        # isequal is total: comparing against a plain array returns false, never throws.
        @test isequal(x, [1.0, 2.0, 3.0]) === false
        @test isequal([1.0, 2.0, 3.0], x) === false
    end

    @testset "Attaching names" begin
        # from_chains(...; flat=true) attaches names to the value itself, not alongside it.
        A = [100c + 10v + i for i in 1:2, v in 1:3, c in 1:4]
        rd = from_chains(A, [:alpha, :beta, :gamma]; flat=true)
        @test variables(rd) == [:alpha, :beta, :gamma]
        @test nchains(rd) == 4
        # Attaching names must not disturb the draws layout.
        @test draws(rd) == draws(from_chains(A))

        # Wrong number of names is rejected.
        @test_throws ErrorException from_chains(A, [:alpha, :beta]; flat=true)
        @test_throws ErrorException from_chains(A, [:alpha, :beta])

        # The RVar constructor takes a names kwarg.
        d = reshape(collect(1.0:12.0), 4, 3)
        x = RVar(d; names=[:a, :b, :c])
        @test variables(x) == [:a, :b, :c]
        @test draws(x) == d

        # names composes with nchains.
        xc = RVar(d; nchains=2, names=[:a, :b, :c])
        @test nchains(xc) == 2
        @test variables(xc) == [:a, :b, :c]

        # names composes with with_chains.
        wc = reshape(collect(1.0:24.0), 2, 4, 3)   # (iterations, chains, 3 vars)
        xw = RVar(wc; with_chains=true, names=[:a, :b, :c])
        @test nchains(xw) == 4
        @test variables(xw) == [:a, :b, :c]

        # Names on a non-vector RV are rejected.
        d2 = reshape(collect(1.0:24.0), 4, 2, 3)
        @test_throws ErrorException RVar(d2; names=[:a, :b])
    end

    @testset "Name-based and names-preserving indexing" begin
        d = reshape(collect(1.0:12.0), 4, 3)
        x = RVar(d; names=[:a, :b, :c])

        # Indexing by name gives the scalar RV for that parameter.
        @test draws(x[:b]) == d[:, 2]
        @test x[:b] isa RVar{Float64, 0}
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
        @test occursin("available: a, b, c", err)

        # Name indexing on an unnamed RV errors rather than returning nonsense.
        u = RVar(d)
        @test_throws ErrorException u[:a]

        # Subsetting an unnamed RV still works and stays unnamed.
        @test variables(u[2:3]) === nothing
        @test draws(u[2:3]) == d[:, 2:3]

        # nchains survives subsetting.
        xc = RVar(d; nchains=2, names=[:a, :b, :c])
        @test nchains(xc[2:3]) == 2
        @test nchains(xc[:b]) == 2
    end

    @testset "Name propagation" begin
        d = reshape(collect(1.0:12.0), 4, 3)
        x = RVar(d; names=[:a, :b, :c])
        y = RVar(d .* 2; names=[:a, :b, :c])
        z = RVar(d .* 3; names=[:p, :q, :r])
        u = RVar(d .* 4)                       # multi-draw, unnamed
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
        A = RVar(randn(4, 2, 3))
        @test variables(A * x) === nothing
        @test variables(vcat(x, x)) === nothing

        # _maybe_names must refuse to attach names whose length no longer matches the
        # broadcast result.
        one_named = RVar(randn(4, 1); names=[:only])
        widened = one_named .+ as_rs([1.0, 2.0, 3.0])
        @test size(widened) == (3,)
        @test variables(widened) === nothing

        # nchains bookkeeping is unaffected by the names plumbing.
        xc = RVar(d; nchains=2, names=[:a, :b, :c])
        @test nchains(xc .+ 1) == 2
        @test variables(xc .+ 1) == [:a, :b, :c]

        # A named single-draw constant keeps its names when no multi-draw operand supplies
        # any, so `+` and `.+` agree.
        sd = RVar(reshape([1.0, 2.0, 3.0], 1, 3); names=[:a, :b, :c])
        @test variables(sd + 1) == [:a, :b, :c]
        @test variables(sd .+ 1) == [:a, :b, :c]
        @test variables(sin(sd)) == [:a, :b, :c]
        @test variables(sin.(sd)) == [:a, :b, :c]
        # A multi-draw operand still wins over a differently-named constant.
        d3 = reshape(collect(1.0:12.0), 4, 3)
        @test variables(RVar(d3; names=[:p, :q, :r]) .+ sd) == [:p, :q, :r]
    end

    @testset "show with names" begin
        d = reshape(collect(1.0:12.0), 4, 3)
        named = RVar(d; names=[:alpha, :beta, :gamma])
        s = sprint(show, named)
        @test occursin("[alpha]", s)
        @test occursin("[beta]", s)
        @test occursin("[gamma]", s)

        # Unnamed values keep the positional labels.
        plain = sprint(show, RVar(d))
        @test occursin("[1]", plain)
        @test !occursin("[alpha]", plain)

        # The REPL path (MIME"text/plain") must show the same summary, not Base's
        # element-by-element array rendering.
        @test occursin("[alpha]", sprint(show, MIME"text/plain"(), named))
        @test occursin("±", sprint(show, MIME"text/plain"(), named))
    end

    @testset "Broadcasting against plain arrays" begin
        # A plain array is draw-invariant: it enters broadcast as (1, shape...) and repeats
        # along the draws axis. Before the backing-array rewrite these all threw.
        d = reshape(collect(1.0:12.0), 4, 3)   # 4 draws, length-3 vector RV
        x = RVar(d)
        p = [10.0, 20.0, 30.0]
        pr = reshape(p, 1, 3)

        @test draws(x .+ p) == d .+ pr
        @test draws(p .+ x) == pr .+ d
        @test draws(x .- p) == d .- pr
        @test draws(x .* p) == d .* pr
        @test draws(p ./ x) == pr ./ d
        @test x .+ p isa RVar{Float64, 1}

        # Non-broadcast operators route through the same path.
        @test draws(x + p) == d .+ pr

        # N = 2, so the logical shape has more than one axis to align.
        D2 = reshape(collect(1.0:24.0), 4, 2, 3)
        M = RVar(D2)
        Q = reshape(collect(1.0:6.0), 2, 3)
        @test draws(M .+ Q) == D2 .+ reshape(Q, 1, 2, 3)

        # Fused expressions mixing an RV, a plain array and a scalar.
        @test draws(x .* p .+ 1) == d .* pr .+ 1

        # Metadata survives the plain-array path.
        @test variables(RVar(d; names=[:a, :b, :c]) .+ p) == [:a, :b, :c]
        @test nchains(RVar(d; nchains=2) .+ p) == 2

        # A single-draw RVar constant still recycles across draws.
        @test draws(x .+ as_rs([100.0, 200.0, 300.0])) == d .+ reshape([100.0, 200.0, 300.0], 1, 3)

        # map with a plain array operand routes into the same machinery.
        @test draws(map(+, x, p)) == d .+ pr

        # Shape mismatches must still be rejected rather than silently recycled.
        @test_throws DimensionMismatch x .+ [1.0, 2.0]
    end

    @testset "rvars regroups flattened array parameters" begin
        # A[i, v, c] carries a distinguishable value per (iteration, variable, chain), so a
        # misplaced element cannot coincide with the right one.
        n_iter, n_chain = 2, 4
        nms = [:sigma,
               Symbol("x[1,1]"), Symbol("x[2,1]"), Symbol("x[1,2]"),
               Symbol("x[2,2]"), Symbol("x[1,3]"), Symbol("x[2,3]"),
               Symbol("b[1]"), Symbol("b[2]")]
        A = float([1000v + 100c + i for i in 1:n_iter, v in 1:length(nms), c in 1:n_chain])

        p = from_chains(A, nms)
        @test p isa NamedTuple
        @test keys(p) == (:sigma, :x, :b)

        # A scalar parameter stays a scalar RV; shapes come from the parsed indices.
        @test p.sigma isa RVar{Float64, 0}
        @test p.x isa RVar{Float64, 2}
        @test size(p.x) == (2, 3)
        @test p.b isa RVar{Float64, 1}
        @test size(p.b) == (2,)

        # Indexing the reassembled array parameter gives a scalar RV over all draws.
        @test p.x[1, 3] isa RVar{Float64, 0}
        @test ndraws(p.x[1, 3]) == n_iter * n_chain

        # Chain bookkeeping survives the regrouping.
        for v in values(p)
            @test nchains(v) == n_chain
            @test niterations(v) == n_iter
            @test ndraws(v) == n_iter * n_chain
        end

        # Every element must hold exactly the draws of its own flat column.
        flat = from_chains(A, nms; flat=true)
        @test draws(p.sigma) == draws(flat[:sigma])
        for t in 1:2, pat in 1:3
            @test draws(p.x[t, pat]) == draws(flat[Symbol("x[$t,$pat]")])
        end
        for k in 1:2
            @test draws(p.b[k]) == draws(flat[Symbol("b[$k]")])
        end

        # Placement is by parsed index, not by column order: shuffling the names must
        # permute the columns and leave the assembled parameter identical.
        perm = [1, 7, 3, 5, 2, 6, 4, 9, 8]
        q = from_chains(A[:, perm, :], nms[perm])
        @test keys(q) == (:sigma, :x, :b)
        @test draws(q.x) == draws(p.x)
        @test draws(q.b) == draws(p.b)

        # Names without a bracketed integer index are left alone as scalars.
        odd = [:sigma, Symbol("x[a]"), Symbol("x[1]")]
        r = from_chains(A[:, 1:3, :], odd)
        @test keys(r) == (:sigma, Symbol("x[a]"), :x)
        @test r.x isa RVar{Float64, 1}
        @test size(r.x) == (1,)
    end

    @testset "rvars rejects malformed parameter names" begin
        A = float([1000v + 100c + i for i in 1:2, v in 1:4, c in 1:4])

        # A gap in the index range would silently produce garbage elements.
        @test_throws ErrorException from_chains(A, [Symbol("x[1]"), Symbol("x[2]"),
                                                   Symbol("x[4]"), :s])
        # Same index twice.
        @test_throws ErrorException from_chains(A, [Symbol("x[1]"), Symbol("x[1]"),
                                                   Symbol("x[2]"), :s])
        # Mixed dimensionality under one base name.
        @test_throws ErrorException from_chains(A, [Symbol("x[1]"), Symbol("x[1,1]"),
                                                   Symbol("x[2]"), :s])
        # A name used both with and without an index.
        @test_throws ErrorException from_chains(A, [:x, Symbol("x[1]"),
                                                   Symbol("x[2]"), :s])
        # Duplicated scalar name.
        @test_throws ErrorException from_chains(A, [:x, :x, :y, :s])
        # Only 1-based indices are supported.
        @test_throws ErrorException from_chains(A, [Symbol("x[0]"), Symbol("x[1]"),
                                                   Symbol("x[2]"), :s])

        # rvars needs names, and needs per-element draws to regroup.
        @test_throws ErrorException rvars(from_chains(A))
        @test_throws ErrorException rvars(RVar(randn(10, 2, 3)))
    end

    @testset "Dimension names and labels" begin
        # a[trial, arm] with 2 trials and 3 arms, plus a scalar.
        n_trial, n_arm = 2, 3
        nms = [[Symbol("a[$i,$j]") for j in 1:n_arm for i in 1:n_trial]..., :s]
        A = float([1000v + 100c + i for i in 1:2, v in 1:length(nms), c in 1:4])
        arms = ["control", "drug", "placebo"]

        p = rvars(A, nms; dims=(a=(:trial, :arm),), labels=(arm=arms,))
        @test dimnames(p.a) == (:trial, :arm)
        @test dimlabels(p.a) == (nothing, arms)
        @test dimlabels(p.a, :arm) == arms
        @test dimlabels(p.a, 2) == arms
        @test dimlabels(p.a, :trial) === nothing
        # Undeclared parameters stay unnamed.
        @test dimnames(p.s) === nothing

        # Label lookup resolves to the same element as the positional index, and accepts
        # either spelling of a string/symbol label.
        @test draws(p.a[trial=1, arm=:drug]) == draws(p.a[1, 2])
        @test draws(p.a[trial=1, arm="drug"]) == draws(p.a[1, 2])
        @test p.a[trial=1, arm=:drug] isa RVar{Float64, 0}
        # Positions still work on a labelled axis.
        @test draws(p.a[trial=1, arm=2]) == draws(p.a[1, 2])

        # A partially indexed RVar keeps the surviving axes, names and labels.
        sl = p.a[arm=:drug]
        @test sl isa RVar{Float64, 1}
        @test size(sl) == (n_trial,)
        @test dimnames(sl) == (:trial,)
        @test draws(sl) == draws(p.a[:, 2])

        vsl = p.a[arm=["drug", "placebo"]]
        @test size(vsl) == (n_trial, 2)
        @test dimnames(vsl) == (:trial, :arm)
        @test dimlabels(vsl) == (nothing, ["drug", "placebo"])

        # Positional slicing carries metadata too, dropping scalar-indexed axes.
        @test dimnames(p.a[:, 2]) == (:trial,)
        @test dimnames(p.a[1, :]) == (:arm,)
        @test dimlabels(p.a[1, :]) == (arms,)
        @test dimnames(p.a[1, 2]) === nothing   # scalar RV has no axes

        # Metadata survives shape-preserving arithmetic, and a metadata-free operand
        # (here a scalar RV) defers rather than forcing it to be dropped.
        for y in (p.a .+ 1, p.a .* 2, -p.a, sin.(p.a), copy(p.a), p.a - p.a, p.a .+ p.s)
            @test dimnames(y) == (:trial, :arm)
        end
        @test dimlabels(p.a .+ 1) == (nothing, arms)
        # Rank-changing operations drop them rather than keeping stale names.
        @test dimnames(reshape(p.a, (6,))) === nothing
        @test dimnames(rs_mean(p.a)) === nothing

        # isequal accounts for dimension metadata.
        plain = rvars(A, nms).a
        @test !isequal(p.a, plain)
        @test isequal(p.a, rvars(A, nms; dims=(a=(:trial, :arm),), labels=(arm=arms,)).a)

        # show reports named axes and uses labels for element rows.
        @test occursin("trial=2,arm=3", sprint(show, p.a))
        @test occursin("[control]", sprint(show, p.a[trial=1]))

        # 3-d parameters and dimension-keyed label sharing.
        nms3 = [Symbol("b[$i,$j,$k]") for k in 1:2 for j in 1:n_arm for i in 1:n_trial]
        A3 = float([1000v + 100c + i for i in 1:2, v in 1:length(nms3), c in 1:4])
        q = rvars(A3, nms3; dims=(b=(:trial, :arm, :time),), labels=(arm=arms,))
        @test dimnames(q.b) == (:trial, :arm, :time)
        @test dimlabels(q.b) == (nothing, arms, nothing)
        @test dimnames(q.b[time=1, arm=:control]) == (:trial,)

        # A single Symbol is accepted for a 1-d parameter.
        nms1 = [Symbol("v[$i]") for i in 1:3]
        A1 = float([1000v + 100c + i for i in 1:2, v in 1:3, c in 1:4])
        r = rvars(A1, nms1; dims=(v=:site,), labels=(site=["x", "y", "z"],))
        @test dimnames(r.v) == (:site,)
        @test draws(r.v[site="y"]) == draws(r.v[2])
    end

    @testset "Dimension metadata rejects bad specs" begin
        n_trial, n_arm = 2, 3
        nms = [[Symbol("a[$i,$j]") for j in 1:n_arm for i in 1:n_trial]..., :s]
        A = float([1000v + 100c + i for i in 1:2, v in 1:length(nms), c in 1:4])
        arms = ["control", "drug", "placebo"]
        p = rvars(A, nms; dims=(a=(:trial, :arm),), labels=(arm=arms,))

        # Wrong number of dimension names for the parameter's rank.
        @test_throws ErrorException rvars(A, nms; dims=(a=(:trial,),))
        @test_throws ErrorException rvars(A, nms; dims=(a=(:t, :u, :v),))
        # Dimension names must be unique.
        @test_throws ErrorException rvars(A, nms; dims=(a=(:trial, :trial),))
        # Naming the dimensions of a scalar parameter.
        @test_throws ErrorException rvars(A, nms; dims=(s=(:x,),))
        # A parameter that isn't in the fit is a typo, not a no-op.
        @test_throws ErrorException rvars(A, nms; dims=(aa=(:trial, :arm),))
        # Label vector must be as long as its axis, and unique.
        @test_throws ErrorException rvars(A, nms; dims=(a=(:trial, :arm),),
                                          labels=(arm=["x", "y"],))
        @test_throws ErrorException rvars(A, nms; dims=(a=(:trial, :arm),),
                                          labels=(arm=["x", "x", "y"],))
        # Labels for columns that are not dimensions are ignored, not an error.
        @test dimnames(rvars(A, nms; dims=(a=(:trial, :arm),),
                             labels=(arm=arms, unrelated=[1, 2])).a) == (:trial, :arm)

        # Unknown dimension name / unknown label at index time.
        @test_throws ErrorException p.a[patient=1]
        @test_throws ErrorException p.a[arm=:sham]
        # An unlabelled axis cannot resolve a label.
        @test_throws ErrorException p.a[trial=:first]
        # Dimension-name indexing needs dimension names.
        @test_throws ErrorException rvars(A, nms).a[trial=1]
        # dims/labels are meaningless for the flat representation.
        @test_throws ErrorException from_chains(A, nms; flat=true, dims=(a=(:trial, :arm),))
    end

    @testset "recover_types" begin
        data = (patient = [2, 1, 3, 1],
                arm     = ["drug", "control", "drug", "control"],
                y       = [0.5, 1.5, 2.5, 3.5])

        labs = recover_types(data)
        # Sorted by default, matching how R factors and CategoricalArrays number levels.
        @test labs.arm == ["control", "drug"]
        @test labs.patient == [1, 2, 3]
        # A continuous column is a measurement, not a dimension.
        @test !haskey(labs, :y)

        @test recover_types(data; sorted=false).arm == ["drug", "control"]

        # Feeds straight into rvars, ignoring columns that aren't dimensions.
        nms = [Symbol("a[$i,$j]") for j in 1:2 for i in 1:3]
        A = float([1000v + 100c + i for i in 1:2, v in 1:6, c in 1:4])
        p = rvars(A, nms; dims=(a=(:patient, :arm),), labels=labs)
        @test dimnames(p.a) == (:patient, :arm)
        @test dimlabels(p.a) == ([1, 2, 3], ["control", "drug"])
        @test draws(p.a[patient=3, arm="drug"]) == draws(p.a[3, 2])
    end

    @testset "gather_draws" begin
        @testset "rank 0, chain/draw packing" begin
            x = RVar(collect(1.0:6.0); nchains=2)  # niterations = 3
            g = gather_draws(x)
            @test keys(g) == (:chain, :draw, :value)
            @test g.chain == [1, 1, 1, 2, 2, 2]
            @test g.draw == [1, 2, 3, 1, 2, 3]
            @test g.value == collect(1.0:6.0)
        end

        @testset "rank 1/2/3 via rvars, round-trip" begin
            n_trial, n_arm = 2, 3
            nms = [[Symbol("a[$i,$j]") for j in 1:n_arm for i in 1:n_trial]..., :s]
            A = float([1000v + 100c + i for i in 1:2, v in 1:length(nms), c in 1:4])
            arms = ["control", "drug", "placebo"]
            p = rvars(A, nms; dims=(a=(:trial, :arm),), labels=(arm=arms,))

            gs = gather_draws(p.s)
            @test keys(gs) == (:chain, :draw, :value)
            @test length(gs.value) == ndraws(p.s)

            ga = gather_draws(p.a)
            @test Set(keys(ga)) == Set((:trial, :arm, :chain, :draw, :value))
            nrows = ndraws(p.a) * length(p.a)
            @test length(ga.value) == nrows
            @test eltype(ga.trial) == Int
            @test eltype(ga.arm) == String
            nit = niterations(p.a)
            d = draws(p.a)
            seen = Set{NTuple{3, Int}}()
            for r in 1:nrows
                tr = ga.trial[r]
                ar = findfirst(==(ga.arm[r]), arms)
                draw_lin = (ga.chain[r] - 1) * nit + ga.draw[r]
                @test ga.value[r] == d[draw_lin, tr, ar]
                push!(seen, (draw_lin, tr, ar))
            end
            # Every (draw, trial, arm) combination appears exactly once.
            @test length(seen) == nrows
            @test seen == Set((i, j, k) for i in 1:ndraws(p.a), j in 1:n_trial, k in 1:n_arm)

            nms3 = [Symbol("b[$i,$j,$k]") for k in 1:2 for j in 1:n_arm for i in 1:n_trial]
            A3 = float([1000v + 100c + i for i in 1:2, v in 1:length(nms3), c in 1:4])
            q = rvars(A3, nms3; dims=(b=(:trial, :arm, :time),), labels=(arm=arms,))
            gb = gather_draws(q.b)
            @test Set(keys(gb)) == Set((:trial, :arm, :time, :chain, :draw, :value))
            @test length(gb.value) == ndraws(q.b) * length(q.b)
        end

        @testset "unnamed dimensions fall back to :dim1, :dim2" begin
            x = RVar(reshape(1.0:24.0, 4, 2, 3))
            g = gather_draws(x)
            @test Set(keys(g)) == Set((:dim1, :dim2, :chain, :draw, :value))
            @test eltype(g.dim1) == Int
            @test eltype(g.dim2) == Int
        end

        @testset "non-String labels preserved (Int, Symbol)" begin
            x = RVar(reshape(1.0:12.0, 2, 2, 3); dimnames=(:idx, :sym),
                     dimlabels=([10, 20], [:a, :b, :c]))
            g = gather_draws(x)
            @test eltype(g.idx) == Int
            @test eltype(g.sym) == Symbol
            @test Set(zip(g.idx, g.sym)) == Set((i, s) for i in (10, 20), s in (:a, :b, :c))
        end

        @testset "flat vector element names are not used as labels" begin
            nms_flat = [:alpha, :beta, :gamma]
            Aflat = float([1000v + 100c + i for i in 1:2, v in 1:3, c in 1:4])
            fv = from_chains(Aflat, nms_flat; flat=true)
            @test dimnames(fv) === nothing
            g = gather_draws(fv)
            @test Set(keys(g)) == Set((:dim1, :chain, :draw, :value))
            @test eltype(g.dim1) == Int
        end

        @testset "collisions with reserved column names" begin
            mk(nm) = RVar(reshape(1.0:8.0, 4, 2); dimnames=(nm,))
            @test_throws ErrorException gather_draws(mk(:chain))
            @test_throws ErrorException gather_draws(mk(:draw))
            @test_throws ErrorException gather_draws(mk(:value))

            xvar = mk(:variable)
            @test gather_draws(xvar) isa NamedTuple  # fine on its own
            @test_throws ErrorException gather_draws((x=xvar,))  # collides once wrapped
        end

        @testset "NamedTuple of RVars: :variable column and ragged dimensions" begin
            n_trial, n_arm = 2, 3
            nms = [[Symbol("a[$i,$j]") for j in 1:n_arm for i in 1:n_trial]..., :s]
            A = float([1000v + 100c + i for i in 1:2, v in 1:length(nms), c in 1:4])
            arms = ["control", "drug", "placebo"]
            p = rvars(A, nms; dims=(a=(:trial, :arm),), labels=(arm=arms,))

            gp = gather_draws(p)
            @test Set(keys(gp)) == Set((:variable, :trial, :arm, :chain, :draw, :value))
            @test Set(unique(gp.variable)) == Set((:a, :s))
            @test length(gp.value) == ndraws(p.a) * length(p.a) + ndraws(p.s)

            s_rows = findall(==(:s), gp.variable)
            a_rows = findall(==(:a), gp.variable)
            @test all(ismissing, gp.trial[s_rows])
            @test all(ismissing, gp.arm[s_rows])
            @test !any(ismissing, gp.trial[a_rows])
            @test !any(ismissing, gp.arm[a_rows])
            @test eltype(gp.trial) == Union{Missing, Int}
            @test eltype(gp.arm) == Union{Missing, String}
        end

        @testset "NamedTuple of RVars: disagreeing labels error" begin
            n_trial, n_arm = 2, 3
            nms = [Symbol("a[$i,$j]") for j in 1:n_arm for i in 1:n_trial]
            A = float([1000v + 100c + i for i in 1:2, v in 1:length(nms), c in 1:4])
            a1 = rvars(A, nms; dims=(a=(:trial, :arm),),
                       labels=(arm=["control", "drug", "placebo"],)).a
            a2 = rvars(A, nms; dims=(a=(:trial, :arm),), labels=(arm=["x", "y", "z"],)).a
            @test_throws ErrorException gather_draws((p1=a1, p2=a2))
            # Agreeing labels (same object built the same way) do not error.
            a3 = rvars(A, nms; dims=(a=(:trial, :arm),),
                       labels=(arm=["control", "drug", "placebo"],)).a
            @test gather_draws((p1=a1, p3=a3)) isa NamedTuple
        end

        @testset "empty NamedTuple errors" begin
            @test_throws ErrorException gather_draws(NamedTuple())
        end

        if HAS_TABLES
            @testset "Tables.jl interface" begin
                x = RVar(reshape(1.0:12.0, 2, 2, 3); dimnames=(:idx, :sym),
                         dimlabels=([10, 20], [:a, :b, :c]))
                g = gather_draws(x)
                @test Tables.istable(typeof(g))
                @test Tables.columntable(g) == g
            end
        end
    end

    if HAS_MCMCCHAINS
        @testset "MCMCChains extension value correctness (H7)" begin
            n_iter, n_var, n_chain = 2, 3, 4
            val = reshape([100c + 10v + i for i in 1:n_iter, v in 1:n_var, c in 1:n_chain],
                          n_iter, n_var, n_chain)
            chn = MCMCChains.Chains(val, [:a, :b, :cc])

            rd = RVar(chn; flat=true)
            @test rd isa RVar{<:Any, 1}
            @test size(rd) == (n_var,)
            @test nchains(rd) == n_chain
            @test niterations(rd) == n_iter

            wc = draws(rd; with_chains=true)
            for i in 1:n_iter, v in 1:n_var, c in 1:n_chain
                @test wc[i, c, v] == val[i, v, c]
            end

            # from_chains(::Chains) is the documented alias for the constructor.
            @test draws(from_chains(chn; flat=true)) == draws(rd)

            # Parameter names come straight off the Chains object.
            @test variables(rd) == [:a, :b, :cc]
            @test variables(from_chains(chn; flat=true)) == [:a, :b, :cc]

            # Names line up with the right columns, so name lookup and position agree.
            for (j, nm) in enumerate([:a, :b, :cc])
                @test draws(rd[nm]) == draws(rd)[:, j]
            end

            # All-scalar parameters: the default result is one scalar RV per parameter.
            p = RVar(chn)
            @test p isa NamedTuple
            @test keys(p) == (:a, :b, :cc)
            for nm in (:a, :b, :cc)
                @test p[nm] isa RVar{<:Any, 0}
                @test draws(p[nm]) == draws(rd[nm])
            end
            @test keys(from_chains(chn)) == keys(p)
            @test keys(rvars(chn)) == keys(p)
        end

        @testset "MCMCChains array parameters become N-d RVars" begin
            # A model declaring x[trial, patient] reaches the chain as one flat column per
            # element; extraction must hand back a 2-d RVar of the declared shape.
            n_iter, n_chain, n_trial, n_patient = 3, 2, 2, 3
            arr_names = [Symbol("x[$t,$pt]") for pt in 1:n_patient for t in 1:n_trial]
            nms = [arr_names..., :sigma]
            n_var = length(nms)
            val = float([1000v + 100c + i for i in 1:n_iter, v in 1:n_var, c in 1:n_chain])
            chn = MCMCChains.Chains(val, nms)

            # Rank and shape are what extraction promises; the element type is whatever
            # MCMCChains stored (it may be a Union{Missing, Float64}), so don't pin it.
            (; x, sigma) = RVar(chn)
            @test x isa RVar{<:Any, 2}
            @test size(x) == (n_trial, n_patient)
            @test sigma isa RVar{<:Any, 0}
            @test nchains(x) == n_chain
            @test niterations(x) == n_iter

            # The first trial and third patient is a scalar RV holding just that element's
            # draws, in (iteration, chain) order.
            e = x[1, 3]
            @test e isa RVar{<:Any, 0}
            @test ndraws(e) == n_iter * n_chain
            v_col = findfirst(isequal(Symbol("x[1,3]")), MCMCChains.names(chn))
            @test draws(e) == vec([val[i, v_col, c] for i in 1:n_iter, c in 1:n_chain])

            # Summaries over draws reduce to plain numbers of the parameter's shape.
            @test mean(x) isa AbstractMatrix
            @test size(mean(x)) == (n_trial, n_patient)
            @test mean(e) ≈ mean(draws(e))
        end

        @testset "MCMCChains dimension names and labels" begin
            n_iter, n_chain, n_trial, n_arm = 20, 2, 2, 3
            nms = [[Symbol("a[$i,$j]") for j in 1:n_arm for i in 1:n_trial]...,
                   [Symbol("b[$i,$j,$k]") for k in 1:2 for j in 1:n_arm for i in 1:n_trial]...,
                   :sigma]
            val = randn(n_iter, length(nms), n_chain)
            chn = MCMCChains.Chains(val, nms)
            arms = ["control", "drug", "placebo"]

            p = RVar(chn; dims=(a=(:trial, :arm), b=(:trial, :arm, :time)),
                          labels=(arm=arms,))
            @test dimnames(p.a) == (:trial, :arm)
            @test dimnames(p.b) == (:trial, :arm, :time)
            # The same dimension is declared once and applies to every parameter using it.
            @test dimlabels(p.a, :arm) == arms
            @test dimlabels(p.b, :arm) == arms
            @test dimnames(p.sigma) === nothing

            @test draws(p.a[trial=2, arm=:placebo]) == draws(p.a[2, 3])
            @test dimnames(p.b[time=1]) == (:trial, :arm)
            @test nchains(p.b[arm=:drug]) == n_chain

            # rvars(::Chains) and from_chains(::Chains) take the same keywords.
            @test dimnames(rvars(chn; dims=(a=(:trial, :arm),)).a) == (:trial, :arm)
            @test dimnames(from_chains(chn; dims=(a=(:trial, :arm),)).a) == (:trial, :arm)
        end
    else
        @info "MCMCChains not available; skipping extension tests (run via Pkg.test)"
    end

end
