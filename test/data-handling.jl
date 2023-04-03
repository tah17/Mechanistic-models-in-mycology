using Test
using Random
using CSV
Random.seed!(26922)
include("../src/ODEs.jl")
include("../src/data-handling.jl")

filepath = joinpath(@__DIR__, "../")
@testset "Data Handling" begin
    test_data = CSV.read(joinpath(filepath, "data/data.csv"), DataFrame)
    #
    # ode arguments for test data sets
    #
    solver = RK4()
    x0 = [1.0, 0.0, 0.0, 0.0]
    t_pts = sort(unique(test_data.Time))
    tspan = (0.0, t_pts[end])
    saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
    ode_species = ["F", "C", "N", "Fd"]
    no_of_species = length(ode_species)
    ODE = TotalDEInput(Example_ODE, tspan, solver, saveat, x0, no_of_species)
    function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
      x[1] < 1e-6
    end
    function affect!(integrator)
       integrator.u[1] = 0
    end
    cb = DiscreteCallback(condition, affect!)
    maxiters = Int64(1e4)
    measured_species = unique(test_data.Species)
    exp_cond = "High IC"
    test_data_high_ic = filter(x -> x.Condition == exp_cond, test_data)
    number_of_exps = length(unique(test_data_high_ic.Experiment))
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
    # tests get_fake_data()
    #
    sim_dat = get_fake_data(test_data, "High IC", ode_species, Example_ODE, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test size(sim_dat)[1] == length(t_pts)
    @test size(sim_dat)[3] == length(ode_species)
    @test size(sim_dat)[2] == number_of_exps
    @test all([sim_dat[:, 1, :] == sim_dat[:, i, :] for i in 1:number_of_exps])
    sim_dat2 = get_fake_data(test_data, "High IC", ode_species, Example_ODE, p, solver, saveat; x0 = x0)   #checks kwargs don't change output type and size
    @test size(sim_dat) == size(sim_dat2)
    sim_dat3 = get_fake_data(test_data, "High IC", ode_species, Example_ODE, p, solver, saveat; x0 = x0, maxiters = maxiters) #also checks kwargs don't change output type and size
    @test size(sim_dat2) == size(sim_dat3)
    sims = simulate(Example_ODE, p, tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, maxiters = maxiters, cb = cb)
    sim_dat_alt = get_fake_data(test_data, "High IC", ode_species, sims, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb) #alt arguments for get_fake_data() where ODE solution directly provided
    @test sim_dat == sim_dat_alt
    sim_dat_alt2 = get_fake_data(Example_ODE, ode_species, p, saveat, t_pts, solver, x0, number_of_exps; maxiters = maxiters, cb = cb) #alt arguments where no real data is provided only time points and number of experiments
    @test sim_dat_alt == sim_dat_alt2
    #
    # tests add_noise()
    #
    test_dict = Dict(measured_species[1] => sim_dat[:, :, 1] + sim_dat[:, :, 4], measured_species[2] => sim_dat[:, :, 2], measured_species[3] =>  sim_dat[:, :, 3])  #converts output of get_fake_data() to Dict() for further use in other functions
    noisy_dict = add_noise(test_dict, "none", test_data, exp_cond)
    @test test_dict == noisy_dict   #we expect no added noise to not change the fake data
    @test typeof(noisy_dict) <: Dict
    #
    #  generate some fake variance and maximum values of data to be used to calibrate noise levels in add_noise()
    #
    df = by(test_data_high_ic, :Species) do df
        DataFrame(m = mean(df[:Reading]), s² = var(df[:Reading]), max = maximum(df[:Reading]))
    end
    var_used = df.s²
    max_used = df.max
    noises = ["none", "low", "med", "high", "max"]
    for noise in noises
        new_noisy_dict = add_noise(test_dict, noise, test_data, exp_cond, RNG = MersenneTwister(29850))
        @test keys(noisy_dict) == keys(new_noisy_dict)
        @test all(size.(values(noisy_dict)) .== size.(values(new_noisy_dict)))
        @test new_noisy_dict == add_noise(test_dict, noise, var_used; data_maxs = max_used, RNG = MersenneTwister(29850))   #checks both input methods give the same output
    end
    noisy_dict2 = add_noise(test_dict, "none", 5*ones(3); data_maxs = 10*ones(3))
    noisy_dict3 = add_noise(test_dict, "med", 1e-6*ones(3))  #tiny noise for testing so readings do not get automatically set to 0.0 (if added noise makes values < 0)
    @test test_dict == noisy_dict2
    noisy_dict4 = add_noise(test_dict, "med", -1*ones(3))
    @test all(ismissing.(vcat([get(noisy_dict4, k, missing) for k in keys(noisy_dict4)]...)))   #tests negative variance values return missing values
    #
    # tests emulate_sparsity()
    #
    test_df = emulate_sparsity(noisy_dict, test_data, exp_cond)
    @test typeof(test_df) <: DataFrame
    @test test_df.Time == test_data_high_ic.Time
    @test test_df.Experiment == test_data_high_ic.Experiment
    @test names(test_df) == names(test_data_high_ic)
    @test size(test_df) == size(test_data_high_ic)
    #
    # tests make_df()
    #
    df_alt = make_df(noisy_dict, t_pts, "High IC")
    @test typeof(df_alt) <: DataFrame
    @test names(test_df) == names(df_alt)
    @test unique(df_alt.Time) == t_pts
    @test length(df_alt.Experiment) == length(t_pts)*number_of_exps*length(measured_species)
    @test length(unique(df_alt.Experiment)) == number_of_exps*length(measured_species)
    @test unique(df_alt.Species) == unique(test_df.Species)
    df_alt2 = make_df(noisy_dict2, t_pts, "High IC", starting_exp = 10)
    @test typeof(df_alt2) <: DataFrame
    @test names(df_alt) == names(df_alt2)
    @test unique(df_alt2.Time) == t_pts
    @test length(df_alt2.Experiment) == length(t_pts)*number_of_exps*length(measured_species)
    @test length(unique(df_alt2.Experiment)) == number_of_exps*length(measured_species)
    @test unique(df_alt2.Species) == unique(test_df.Species)
    @test unique(df_alt2.Experiment) == ["E$i" for i in 11:length(unique(df_alt.Experiment))+10]
    df_alt3 = make_df(noisy_dict3, t_pts, "High IC")
    #
    # tests residuals()
    #
    # user defined function that takes in the fake data set (dataframe), the experimental condition e.g. "High IC" and the ODE arguments needed to generate a fake data set.
    # and outputs a dictionary that has values of 2D arrays of data (time pts x no of exps) and keys that are the measured species name (String).
    #
    function user_defined(real_data::DataFrame,
                          condition::String,
                          ode_species::Array{String,1},
                          Example_ODE::Function,
                          p_draw::AbstractArray{Float64,1},
                          solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                          saveat::Real;
                          x0::Union{AbstractArray{Float64,1}, Missing} = nothing,
                          maxiters::Union{Int64, Nothing} = nothing,
                          cb::Union{Nothing, Any} = nothing)

        measured_species = unique(real_data.Species)
        sim_dat = get_fake_data(real_data, condition, ode_species, Example_ODE, p, solver, saveat;
                                x0 = x0, maxiters = maxiters, cb = cb)
        f_all = sim_dat[:, :, 1] + sim_dat[:, :, 4]  #Fall = F + Fd
        c = sim_dat[:, :, 2]
        n = sim_dat[:, :, 3]
        return Dict(measured_species[1] => f_all, measured_species[2] => c, measured_species[3] =>  n)
    end
    sim_dat_alt3 = user_defined(test_data, "High IC", ode_species, Example_ODE, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test noisy_dict == sim_dat_alt3   #noisy dict had no added noise
    res = residuals(Example_ODE, user_defined, test_data, ode_species, p, tspan, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test typeof(res) <: Real
    @test length(res) == 1
    #
    #  fake data generated with parameter p and no added noise (df_alt2) should give a residual of 0 when using the parameter set p
    #
    res_zero = residuals(Example_ODE, user_defined, df_alt2, ode_species, p, tspan, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test res_zero == 0
    #
    #  fake data created with ascending noise
    #
    res_arr = zeros(length(noises))
    for (idx, noise) in enumerate(noises)
        tmp1 = add_noise(test_dict, noise, test_data, exp_cond, RNG = MersenneTwister(29850))
        tmp2 = make_df(tmp1, t_pts, "High IC")
        res_arr[idx] = residuals(Example_ODE, user_defined, tmp2, ode_species, p, tspan, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    end
    @test res_arr[1] == 0   #no noise should give sum of squared errors of 0 if the original parameter set p
    for i in 1:length(res_arr)-1   #check that noises added return ascending residuals
        @test res_arr[i] < res_arr[i+1]
    end
    k = length(df_alt3.Reading) - 1
    tst = rand(Chisq(k))
    noisy_res = residuals(Example_ODE, user_defined, df_alt3, ode_species, vcat(p, 1e-6*ones(length(var_used))), tspan, solver, saveat; x0 = x0, noise = true, maxiters = maxiters, cb = cb)
    @test typeof(noisy_res) <: Real
    @test isapprox(noisy_res, tst, rtol = 0, atol = sqrt(2*k))  #check our function outputs something within a s.d of the expected result i.e. feasibly from a Chisq(k) distribution
    #
    # tests argument errors are caught
    #
    @test_throws ArgumentError get_fake_data(test_data, "High IC", ode_species[1:end-1], Example_ODE, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test_throws ArgumentError get_fake_data(test_data, "hgpweihgpweg", ode_species, Example_ODE, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test_throws ArgumentError get_fake_data(test_data, "fhewufhwlk", ode_species, sims, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
    @test_throws ArgumentError add_noise(test_dict, "hflwejhflwef", test_data, exp_cond)
    @test_throws ArgumentError add_noise(test_dict, "none", test_data, "hfjhfjsf")
    @test_throws ArgumentError add_noise(test_dict, "max", 5*ones(3))
    @test_throws ArgumentError emulate_sparsity(noisy_dict, test_data, "hfoeuhfowef")
    @test_throws ArgumentError emulate_sparsity(filter(x -> first(x) !== "N", noisy_dict), test_data, exp_cond)
    @test_throws ArgumentError residuals(Example_ODE, user_defined, test_data, ode_species, vcat(p, 0 .* var_used), tspan, solver, saveat; x0 = x0, noise = true, maxiters = maxiters, cb = cb)
end
