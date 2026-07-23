using RandomDraws
using Statistics
using LinearAlgebra
using Test

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

end
