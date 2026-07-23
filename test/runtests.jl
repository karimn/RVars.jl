using RandomDraws
using Statistics
using LinearAlgebra
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

@testset "RandomDraws.jl" begin

    @testset "Constructors" begin
        x = RandomDraw(randn(1000))
        @test x isa RandomDraw{Float64, 0}
        @test ndraws(x) == 1000
        @test nchains(x) == 1
        @test length(x) == 1

        y = RandomDraw(randn(1000, 3))
        @test y isa RandomDraw{Float64, 1}
        @test size(y) == (3,)
        @test length(y) == 3

        z = RandomDraw(randn(1000, 4, 3))
        @test z isa RandomDraw{Float64, 2}
        @test size(z) == (4, 3)

        w = RandomDraw(randn(800, 3), nchains=4)
        @test nchains(w) == 4
        @test niterations(w) == 200
        @test ndraws(w) == 800
    end

    @testset "with_chains" begin
        data = randn(200, 3)
        x = RandomDraw(data; with_chains=true)
        @test nchains(x) == 3
        @test niterations(x) == 200
        @test ndraws(x) == 600
        @test size(x) == ()
    end

    @testset "as_rs" begin
        x = as_rs([1.0, 2.0, 3.0])
        @test x isa RandomDraw{Float64, 1}
        @test ndraws(x) == 1
        @test size(x) == (3,)

        s = as_rs(5.0)
        @test s isa RandomDraw{Float64, 0}
        @test ndraws(s) == 1

        m = as_rs(ones(2, 3))
        @test m isa RandomDraw{Float64, 2}
        @test size(m) == (2, 3)
    end

    @testset "AbstractArray" begin
        x = RandomDraw(randn(1000, 3))
        @test size(x) == (3,)

        el = x[1]
        @test el isa RandomDraw{Float64, 0}
        @test ndraws(el) == 1000

        @test length(x) == 3
        @test axes(x) == (Base.OneTo(3),)
        @test ndims(x) == 1
    end

    @testset "Arithmetic" begin
        x = RandomDraw(rand(1000))
        y = RandomDraw(rand(1000))
        @test x + y isa RandomDraw
        @test x - y isa RandomDraw
        @test x * y isa RandomDraw
        @test x / y isa RandomDraw
        @test x + 1.0 isa RandomDraw
        @test 1.0 + x isa RandomDraw
        @test x - 1.0 isa RandomDraw
        @test 1.0 - x isa RandomDraw
        @test x * 2.0 isa RandomDraw
        @test 2.0 * x isa RandomDraw
        @test x / 2.0 isa RandomDraw
        @test 2.0 \ x isa RandomDraw
        @test -x isa RandomDraw
        @test x ^ 2 isa RandomDraw
        for f in [sin, cos, tan, exp, log, abs, sqrt, floor, ceil, round, sign]
            @test f(x) isa RandomDraw
        end
        @test x .+ y isa RandomDraw
        x_vec = RandomDraw(rand(1000, 3))
        @test (x_vec .> 0.5) isa RandomDraw{Bool, 1}
    end

    @testset "Stats over draws" begin
        x = RandomDraw(randn(1000, 3))
        @test mean(x) isa Vector{Float64}
        @test length(mean(x)) == 3
        @test std(x) isa Vector{Float64}
        @test var(x) isa Vector{Float64}
        @test median(x) isa Vector{Float64}

        x_scalar = RandomDraw(randn(1000))
        @test mean(x_scalar) isa Float64
        @test std(x_scalar) isa Float64
        @test var(x_scalar) isa Float64
        @test median(x_scalar) isa Float64
    end

    @testset "E and Pr" begin
        x = RandomDraw(randn(10000, 3))
        @test E(x) ≈ mean(x)
        gt = x .> 0
        @test gt isa RandomDraw{Bool, 1}
        prob = mean(gt)
        @test prob isa Vector{Float64}
        @test all(0.4 .< prob .< 0.6)
    end

    @testset "rs_ summaries" begin
        x = RandomDraw(randn(1000, 4, 3))
        @test rs_mean(x) isa RandomDraw{Float64, 0}
        @test ndraws(rs_mean(x)) == 1000
        @test rs_sum(x) isa RandomDraw
        @test rs_sd(x) isa RandomDraw
        @test rs_var(x) isa RandomDraw
        @test rs_median(x) isa RandomDraw
        @test rs_min(x) isa RandomDraw
        @test rs_max(x) isa RandomDraw

        y = RandomDraw(randn(1000))
        @test rs_quantile(y, [0.25, 0.5, 0.75]) isa RandomDraw{Float64, 1}
    end

    @testset "Constants" begin
        y = RandomDraw(randn(1000))
        z = as_rs(1.0) + y
        @test ndraws(z) == 1000

        c = as_rs(ones(3))
        z2 = c + RandomDraw(randn(1000, 3))
        @test ndraws(z2) == 1000
        @test size(z2) == (3,)

        z3 = y + 1.0
        @test ndraws(z3) == 1000
        m = mean(z3)
        @test m isa Float64
    end

    @testset "rvar_rng" begin
        x = rvar_rng(randn, 3)
        @test x isa RandomDraw{Float64, 1}
        @test size(x) == (3,)
        @test ndraws(x) == 2000
    end

    @testset "from_chains (MCMCChains interop)" begin
        data = randn(200, 5, 4)
        rd = from_chains(data)
        @test rd isa RandomDraw{Float64, 1}
        @test size(rd) == (5,)
        @test ndraws(rd) == 800
        @test nchains(rd) == 4
        @test niterations(rd) == 200

        rd2, names = from_chains(data, [:mu, :sigma, :alpha, :beta, :lp])
        @test rd2 isa RandomDraw
        @test names == [:mu, :sigma, :alpha, :beta, :lp]

        rd3, names3 = from_chains(data, ["mu", "sigma", "alpha", "beta", "lp"])
        @test rd3 isa RandomDraw
        @test names3 == [:mu, :sigma, :alpha, :beta, :lp]
    end

    @testset "Matrix multiplication" begin
        A = RandomDraw(randn(1000, 2, 3))
        B = RandomDraw(randn(1000, 3, 4))
        C = A * B
        @test C isa RandomDraw{Float64, 2}
        @test size(C) == (2, 4)
        @test ndraws(C) == 1000

        v = RandomDraw(randn(1000, 3))
        w = RandomDraw(randn(1000, 3))
        d = dot(v, w)
        @test d isa RandomDraw{Float64, 0}

        M = RandomDraw(randn(1000, 3, 2))
        v2 = RandomDraw(randn(1000, 2))
        mv = M * v2
        @test mv isa RandomDraw{Float64, 1}
        @test size(mv) == (3,)
    end

    @testset "Type stability" begin
        x = RandomDraw(rand(1000))
        y = x + x
        @test y isa RandomDraw{Float64}
    end

    @testset "nchains propagation (H5/H6/M6)" begin
        x = RandomDraw(randn(800, 3), nchains=4)
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
        y = RandomDraw(randn(800, 3), nchains=2)
        @test nchains(x + y) == 1

        # H6: nchains kwarg must be honored for flat-vector input.
        v = RandomDraw(collect(1.0:800.0); nchains=4)
        @test nchains(v) == 4
        @test niterations(v) == 200

        # M6: a single-draw object cannot claim multiple chains.
        @test_throws ErrorException RandomDraw(randn(1, 3); nchains=2)
    end

    @testset "Indexing value correctness (C2/H1/H2)" begin
        nd = 5
        # store[k, i, j] distinguishable per (draw, row, col); visible shape (2, 3).
        store = [1000k + 10i + j for k in 1:nd, i in 1:2, j in 1:3]
        x = RandomDraw(store)
        @test x isa RandomDraw{Int, 2}
        @test size(x) == (2, 3)
        @test length(x) == 6

        # Cartesian access returns the correct element's draws (H1: column-major).
        for i in 1:2, j in 1:3
            el = x[i, j]
            @test el isa RandomDraw{Int, 0}
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
        v = RandomDraw(randn(50, 3))          # N=1
        @test_throws BoundsError v[4]
        @test_throws BoundsError v[100]

        # Materializing elements into scalar RVs via a comprehension round-trips.
        c = [x[i, j] for i in 1:2, j in 1:3]
        @test c isa Matrix{<:RandomDraw}
        @test draws(c[2, 3]) == store[:, 2, 3]

        # Logical indexing selects the right columns (flattened, column-major).
        mask = [true false true; false true false]   # shape (2,3)
        sel = x[mask]
        @test sel isa RandomDraw{Int, 1}
        @test size(sel) == (3,)
        expected_cols = findall(vec(mask))
        flat = reshape(store, nd, 6)
        @test draws(sel) == flat[:, expected_cols]

        # setindex! (scalar) writes across all draws of one element.
        y = RandomDraw(zeros(nd, 2, 3))
        y[li[2, 3]] = 7.0
        @test all(draws(y[2, 3]) .== 7.0)
        @test all(draws(y[1, 1]) .== 0.0)
        @test_throws BoundsError (y[7] = 1.0)
    end

    @testset "Broadcasting over N>=2 RVs (H3)" begin
        nd = 8
        store = reshape(collect(1.0:(nd * 6)), nd, 2, 3)
        x = RandomDraw(store)

        s = sin.(x)
        @test s isa RandomDraw{Float64, 2}
        @test size(s) == (2, 3)
        @test draws(s[1, 2]) == sin.(store[:, 1, 2])

        a = x .+ 0.0
        @test draws(a[2, 3]) == store[:, 2, 3]

        t = x .+ x
        @test draws(t[2, 1]) == store[:, 2, 1] .+ store[:, 2, 1]

        # Broadcasting a scalar-RV against a vector-RV expands the singleton.
        c = as_rs(10.0)                        # N=0, 1 draw
        w = RandomDraw(randn(nd, 3))
        cw = c .+ w
        @test cw isa RandomDraw{Float64, 1}
        @test size(cw) == (3,)
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

    if HAS_MCMCCHAINS
        @testset "MCMCChains extension value correctness (H7)" begin
            n_iter, n_var, n_chain = 2, 3, 4
            val = reshape([100c + 10v + i for i in 1:n_iter, v in 1:n_var, c in 1:n_chain],
                          n_iter, n_var, n_chain)
            chn = MCMCChains.Chains(val, [:a, :b, :cc])

            rd = RandomDraw(chn)
            @test rd isa RandomDraw{<:Any, 1}
            @test size(rd) == (n_var,)
            @test nchains(rd) == n_chain
            @test niterations(rd) == n_iter

            wc = draws(rd; with_chains=true)
            for i in 1:n_iter, v in 1:n_var, c in 1:n_chain
                @test wc[i, c, v] == val[i, v, c]
            end

            # from_chains(::Chains) is the documented alias for the constructor.
            @test draws(from_chains(chn)) == draws(rd)
        end
    else
        @info "MCMCChains not available; skipping extension tests (run via Pkg.test)"
    end

end
