using Test
using Random
using Distributed
using Distributions
include("../src/coverage.jl")
Random.seed!(1234)

@testset "Coverage" begin
    #simple dummy normal distribution at 0 (coverage test)
    niter = 200  #no of simulations per experiment
    n = convert(Int64, 1e4)
    theta_true = [1.0, 0.5]
    theta_samples = hcat([rand(Normal(theta_true[1], theta_true[2]), n) for x in 1:niter]...)
    m = mean(theta_samples, dims = 1)
    s_squared = var(theta_samples, dims = 1)  #sample var
    FIM_estimate = vcat(1/sqrt(n).*sqrt.(s_squared), (sqrt(2)/sqrt(n)).*s_squared)  #set up wald approximation CIs for sample mean and var..
                                                                                    #...https://en.wikipedia.org/wiki/Normal_distribution#Estimation_of_parameters
    alphas = collect(0.01:0.05:0.99)
    theta_estimates = vcat(m, sqrt.(s_squared))
    CIs = Array{Tuple{Float64,Float64},3}(undef, length(theta_true), length(alphas), niter)
    for j in 1:niter
        for (idx, alpha) in enumerate(alphas)
            CIs[:, idx, j] = tuple.(theta_estimates[:, j] - (abs(quantile(Normal(), alpha/2)) .* FIM_estimate[:, j]), theta_estimates[:, j] + (abs(quantile(Normal(), alpha/2)) .* FIM_estimate[:, j]))  #create CIs from the FIMs
        end
    end
    coverages, coverage_cis, confidences = coverage(theta_true, CIs, alphas)  #returns estimated coverage, the coverage CIs and the 1 - alpha values.
    @test sum(ininterval.(confidences, coverage_cis[1, :]))/length(ininterval.(confidences, coverage_cis[1, :])) > 0.9  #0.9 is chosen since that is the expected coverage of the coverage CIs
    @test sum(ininterval.(confidences, coverage_cis[2, :]))/length(ininterval.(confidences, coverage_cis[2, :])) > 0.9
    @test all(ininterval.(coverages[1, :], coverage_cis[1, :]))  #test binomial CIs of coverages are well posed
    @test all(ininterval.(coverages[2, :], coverage_cis[2, :]))
    @test size(coverages) == size(coverage_cis)
    @test size(coverages)[1] == length(theta_true)
    @test size(coverages)[2] == length(alphas) + 2
    @test confidences[2:end-1] == 1 .- alphas
    @test vcat(confidences[1], confidences[end]) == [1.0, 0.0]
    @test vcat(coverage_cis[:, 1], coverage_cis[:, end]) == repeat([(1.0, 1.0), (0.0, 0.0)], inner = 2)
    @test ininterval(1, (0, 3))
    @test !ininterval(1, (-5, 0))
    #
    # test ArgumentErrors are thrown
    #
    @test_throws ArgumentError ininterval(5, (10, 4))
    @test_throws ArgumentError ininterval(5, (10.0, 4))
    @test_throws ArgumentError coverage([theta_true[1]], CIs, alphas)
    @test_throws ArgumentError coverage(theta_true, CIs, alphas[1:end-1])
end
