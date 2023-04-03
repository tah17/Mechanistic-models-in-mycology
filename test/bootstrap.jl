using Test
using SharedArrays
using PyCall
using Random
using Distributions
using CSV
Random.seed!(26922)

include("../src/bootstrap.jl")
include("../src/data-handling.jl")

@testset "Bootstrapping" begin
    #
    # arguments for fake data generation
    #
    solver = RK4()
    x0 = [1.0, 0.0, 0.0, 0.0]
    t_pts = collect(1:10.0)
    tspan = (0.0, t_pts[end])
    number_of_exps = 10
    saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
    ode_species = ["F", "C", "N", "Fd"]
    maxiters = Int64(1e4)
    p = rand.([Truncated(Normal(0.28, 0.1), 0, 2),
           Truncated(Normal(1.2, 0.5), 0, 10),
           Truncated(Normal(1.2, 0.5), 0, 10),
           Truncated(Normal(0.114, 0.05), 0, 10),
           Normal(0.066, 0.03),
           Truncated(Normal(0.225, 0.1), 0, 2),
           Truncated(Normal(0.31, 0.15), 0, 4),
           Truncated(Normal(0.096, 0.045), 0, 2),
           Truncated(Normal(0.09, 0.045), 0, 2)])
    #
    # generate a fake data set
    #
    sim_dat =  get_fake_data(Example_ODE, ode_species, p, saveat, t_pts, solver, x0, number_of_exps; maxiters = maxiters)
    measured_species = ["F_all", "C", "N"]
    test_dict = Dict(measured_species[1] => sim_dat[:, :, 1] + sim_dat[:, :, 4], measured_species[2] => sim_dat[:, :, 2], measured_species[3] =>  sim_dat[:, :, 3])
    test_data = make_df(test_dict, t_pts, "High IC")
    RNG = MersenneTwister(26922)
    #
    # checking boostrap sampling
    #
    bootstrap_data = bootstrap_sample(test_data, RNG)
    @test names(bootstrap_data) == names(test_data)
    @test length(bootstrap_data.Experiment) == length(test_data.Experiment)
    #
    # checking interval creation
    #
    alpha = 0.05  #confidence level (95%)
    n = convert(Int64, 1e4)
    niter = 200  #number of "experiments"
    theta_true = [1.0, 0.5]  #true parameters
    theta_samples = hcat([rand(Normal(theta_true[1], theta_true[2]), n) for x in 1:niter]...)  #sample fake data
    m = mean(theta_samples, dims = 1)
    s_squared = var(theta_samples, dims = 1)  #sample var
    theta_estimates = vcat(m, sqrt.(s_squared))   #MLE estiamtes of mean and standard deviation
    for type in ["percentile", "z-interval", "t-interval", "corrected-t-interval"]   #check that intervals created include the true value
        @test all(ininterval.(theta_true, bootstrap_intervals(theta_estimates, alpha; type = type, theta_hat = theta_true)))
    end
    #
    # tests argument errors are caught
    #
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "hlfsjhfdlf")
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "z-interval")
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "t-interval")
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "corrected-t-interval")
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "z-interval", theta_hat = [theta_true[1]])
    @test_throws ArgumentError bootstrap_intervals(reshape(theta_estimates[1, :], (1, length(theta_estimates[1, :]))), alpha; type = "z-interval", theta_hat = theta_true)
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "t-interval", theta_hat = [theta_true[1]])
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, alpha; type = "corrected-t-interval", theta_hat = [theta_true[1]])
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, 1.5)
    @test_throws ArgumentError bootstrap_intervals(theta_estimates, -1.5)
end
