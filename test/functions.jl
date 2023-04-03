using Test
using DifferentialEquations
using Random
Random.seed!(26922)
include("../src/functions.jl")
include("../src/ODEs.jl")

filepath = joinpath(@__DIR__, "../")
@testset "Main Functions" begin
    #get data from excel sheets
    xf = XLSX.readxlsx(joinpath(filepath, "data/IFNg.xlsx"))["Sheet1"][:]
    wanted_idxs = [1:2, 5:6]  #specify the columns you want from the excel sheet
    dat = get_data_per_species(xf, wanted_idxs, 21)  #reads from rows 21 onwards
    @test size(dat) == (17, 4)
    @test typeof(dat) == Array{Union{Missing, Float64},2}
    @test repeat(dat[:, 1], 1, length(dat[1, :]) - 1) != hcat([dat[:, x] for x in 2:length(dat[1, :])]...)
    xf2 = XLSX.readxlsx(joinpath(filepath, "data/IL-8.xlsx"))["Sheet1"][:]
    wanted_idxs2 = [1:10, 13:16]
    dat2 = get_data_per_species(xf2, wanted_idxs2, 19)
    xf3 = XLSX.readxlsx(joinpath(filepath, "data/N data.xlsx"))["Sheet1"][:]
    wanted_idxs3 = [1:9, 12:15]
    dat3 = get_data_per_species(xf3, wanted_idxs3, 2)
    xf4 = XLSX.readxlsx(joinpath(filepath, "data/TNF alpha.xlsx"))["Sheet1"][:]
    wanted_idxs4 = [1:16, 19:25]
    dat4 = get_data_per_species(xf4, wanted_idxs4, 19)
    xf_tot = [xf, xf2, xf3, xf4]  #list of all excel sheets
    wanted_idxs_tot = [wanted_idxs, wanted_idxs2, wanted_idxs3, wanted_idxs4]  #columns
    start_of_data_tot = [21, 19, 2, 19]  #rows
    dat_tot = get_data(xf_tot, wanted_idxs_tot, start_of_data_tot)
    #
    # checks that get_data() concatenates the data from each spreadsheet correctly
    #
    @test collect(skipmissing(dat[:, 2:end])) == collect(skipmissing(dat_tot[:, 2:end, 1]))
    @test collect(skipmissing(dat2[:, 2:end])) == collect(skipmissing(dat_tot[:, 2:end, 2]))
    @test collect(skipmissing(dat3[:, 2:end])) == collect(skipmissing(dat_tot[:, 2:end, 3]))
    @test collect(skipmissing(dat4[:, 2:end])) == collect(skipmissing(dat_tot[:, 2:end, 4]))
    #
    #ODE arguments
    #
    Tspan = (dat_tot[1, 1, 1], dat_tot[end, 1, 1])
    x0 = [0.5]  #initial conditions
    solver = RK4()
    saveat = 0.01
    no_of_species = 1
    p = [1.0]
    p_σ = [1.0, 0.1]  #with noise
    p_σ_more = [1.0, 1.0, 1.0]  #with more noise
    p_x_0 = [0.5, 1.0]
    @test ismissing(simulate(sigmoid, vcat(p, -1.0), Tspan, solver, saveat; no_of_species = 1, x0 = x0, noise = true))  #negative noise should return missing
    simulations = simulate(sigmoid, p, Tspan, solver, saveat; no_of_species = no_of_species, x0 = x0)
    @test simulations.retcode == :Success
    simulations_x0 = simulate(sigmoid, p_x_0, Tspan, solver, saveat; no_of_species = no_of_species)  #check simulate() works if x0 provided as parameters
    @test simulations.u == simulations_x0.u
    simulations_max_iters = simulate(sigmoid, p, Tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, maxiters = 300)  #check maxiters specification correct
    @test simulations_max_iters.destats.nreject + simulations_max_iters.destats.naccept <= 300
    simulations_more_noise = simulate(sigmoid, p_σ_more, Tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, noise = true)
    @test simulations_more_noise.retcode == :Success
    x0 = missing  #we treat the ICs as parameters
    t_pts = [24.0, 48.0, 72.0]
    tspan = (0.0, t_pts[end])
    saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
    ode_species = ["F", "C", "N", "Fd"]
    no_of_species = length(ode_species)
    p_new = [1.0, 0.0, 0.0, 0.0, 0.27, 1.2, 1.2, 0.114, 0.066, 0.25, 0.31, 0.096, 0.09, 1.0, 1.0, 1.0, 0.0]  #nominal parameter values
    function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
      x[1] < 1e-6
    end
    function affect!(integrator)
       integrator.u[1] = 0
    end
    cb = DiscreteCallback(condition, affect!)
    simulations_ex = simulate(Example_ODE, p_new, tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, maxiters = 300, cb = cb)
    @test simulations_ex.retcode == :Success
    @test all(map(x -> x .>= 0, simulations_ex[1, :]))  #check all species are positive as they represent biological entities
    noise_simulations_ex = simulate(Example_ODE, p_new, tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, noise = true, maxiters = 300, cb = cb)
    @test noise_simulations_ex.retcode == :Success
    @test simulations_ex.u != noise_simulations_ex.u
    #
    # tests argument errors are caught
    #
    @test_throws ArgumentError simulate(sigmoid, p, Tspan, solver, saveat)  #no number of species
    @test_throws ArgumentError simulate(sigmoid, p, Tspan, solver, saveat; no_of_species = 10, x0 = x0)   #wrong no of species with x0 provided
    @test_throws ArgumentError simulate(sigmoid, vcat(p, -1.0), Tspan, solver, saveat; no_of_species = 10, x0 = x0, noise = true)  #wrong no of species with x0 provided and noise specified
    @test_throws ArgumentError simulate(sigmoid, p_x_0, Tspan, solver, saveat; no_of_species = 10)  #wrong no of species with no xo provided
end
