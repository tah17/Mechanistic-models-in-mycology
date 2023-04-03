#
#  pick-params.jl will sample 5 parameter sets for the entire analysis to be run for and saves them in the file data/parameters.csv
#  The parameter sets need result in a solveable ODE and ensure the healthy steady state of the ODE is stable.
#
using Random
using DelimitedFiles
using Distributions
using DataFrames
using CSV
Random.seed!(96421658521890787)
include("../../src/ODEs.jl")
include("../../src/data-handling.jl")
using PyPlot; ENV["MPLBACKEND"]="tkagg"; pygui(true);
filepath = joinpath(@__DIR__, "../..")

#
# read in real data
#
real_data = CSV.read("$filepath/data/data.csv", DataFrame)
#
# ODE arguments
#
solver = RK4()
x0 = missing  #we treat the ICs as parameters
ode_species = ["F", "C", "N", "Fd"]
t_pts = sort(unique(real_data.Time))
tspan = (0.0, t_pts[end])  #timespan
saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
no_of_species = 4
ODE = TotalDEInput(Example_ODE, tspan, solver, saveat, x0, no_of_species)
function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
  x[1] < 1e-6
end
function affect!(integrator)   #sets fungal burden to 0 if condition (fungal burden < 1 CEs) is met
   integrator.u[1] = 0
end
cb = DiscreteCallback(condition, affect!)
no_of_p_sets = 5  #number of parameter sets to do the whole analysis on
no_of_params = 14  #5 ics as there are two fungal i.c.s for different data sets & 9 kinetic rate parameters
maxiters = Int64(1e4)
total_p_draws = Array{Float64,2}(undef, no_of_p_sets, no_of_params)
for i in 1:no_of_p_sets     #find parameter values that are solveable and also satisfy biologically motivated bounds
    p_draw_obs = missing    #simulator will return missing if the ODE cannot be solved for this parameter set
    p_draw = rand.([Normal(4, 0.5),    #normal distributions for random draws of parameters are centred around values in prev. published paper (Tanaka et al. 2015)
                    Normal(7, 0.5),
                    Truncated(Normal(0, 1), 0, 10),
                    Truncated(Normal(0, 1), 0, 10),
                    Truncated(Normal(0, 1), 0, 10),
                    Truncated(Normal(0.28, 0.1), 0, 2),
                    Truncated(Normal(1.2, 0.5), 0, 10),
                    Truncated(Normal(1.2, 0.5), 0, 10),
                    Truncated(Normal(0.114, 0.05), 0, 10),
                    Normal(0.066, 0.03),
                    Truncated(Normal(0.225, 0.1), 0, 2),
                    Truncated(Normal(0.31, 0.15), 0, 4),
                    Truncated(Normal(0.096, 0.045), 0, 2),
                    Truncated(Normal(0.09, 0.045), 0, 2)])
    while ismissing(p_draw_obs) || (p_draw[10]*p_draw[7] <= p_draw[12]*p_draw[11])   #from bifurcation analysis - want N = 0 and C = 0 to be stable steady state.
        p_draw = rand.([Normal(4, 0.5),
                        Normal(7, 0.5),
                        Truncated(Normal(0, 1), 0, 10),
                        Truncated(Normal(0, 1), 0, 10),
                        Truncated(Normal(0, 1), 0, 10),
                        Truncated(Normal(0.28, 0.14), 0, 2),
                        Truncated(Normal(1.2, 0.5), 0, 10),
                        Truncated(Normal(1.2, 0.5), 0, 10),
                        Truncated(Normal(0.114, 0.05), 0, 10),
                        Normal(0.066, 0.03),
                        Truncated(Normal(0.225, 0.1), 0, 2),
                        Truncated(Normal(0.31, 0.15), 0, 4),
                        Truncated(Normal(0.096, 0.04), 0, 2),
                        Truncated(Normal(0.09, 0.045), 0, 2)])
        global p_used = vcat([10^p_draw[i]/10^6 for i in 1:5], p_draw[6:end])
        p_draw_obs = simulate(Example_ODE, p_used[2:end], tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, maxiters = maxiters, cb = cb)
    end
    total_p_draws[i, :] = p_used
end
#
# user defined function that takes in the real data set (dataframe), the experimental condition e.g. "High IC" and the ODE arguments needed to generate a fake data set
# and outputs a dictionary that has values of 2D arrays of data (time pts x no of exps) and keys that are the measured species name (String).
# This function is then passed to all downstream data handling functions so it's important to keep the input and output structure the same.
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

    if condition == "High IC"
        p = p_draw[2:end]
    else
        p = vcat(p_draw[1], p_draw[3:end])
    end
    measured_species = unique(real_data.Species)
    sim_dat = get_fake_data(real_data, condition, ode_species, Example_ODE, p, solver, saveat;
                            x0 = x0, maxiters = maxiters, cb = cb)
    f_all = sim_dat[:, :, 1] + sim_dat[:, :, 4]  #Fall = F + Fd
    c = sim_dat[:, :, 2]
    n = sim_dat[:, :, 3]
    if condition == "High IC"
        return Dict(measured_species[1] => f_all, measured_species[2] => c, measured_species[3] =>  n)
    elseif condition == "Low IC"
        return Dict(measured_species[1] => f_all)
    else
        throw(ArgumentError("$condition is not a valid experimental condition"))
    end
end
#
# visualise
#
noise = true
noise_level = "low"
colour_palette = ["darkgreen", "grey", "navy"]
plt.figure(figsize = (15,6))
for i in 1:no_of_p_sets
    #
    # generate the fake data
    #
    dfs = [DataFrame() for elt in unique(real_data.Condition)]
    for (idx, elt) in enumerate(unique(real_data.Condition))
        sim_dat = user_defined(real_data, elt, ode_species, Example_ODE, total_p_draws[i, :], solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
        if noise
            sim_dat = add_noise(sim_dat, noise_level, real_data, elt)
        end
        dfs[idx] = emulate_sparsity(sim_dat, real_data, elt)
    end
    df = join([df for df in dfs]..., kind  = :outer, on = intersect([names(df) for df in dfs]...))
    df.Reading .= df.Reading .* 10^3
    df = filter(x -> !ismissing(x.Reading), df)
    #
    # plot the fake data
    #
    for (idy, elty) in enumerate(groupby(df, :Condition))
        plot_no = (i + (idy-1)*5)   #ensures that row is by coniditon and columns are by parameter set
        if plot_no == 10
            plot_no = 0
        end
        plt.subplot("25$plot_no")
        for (idx, elt) in enumerate(groupby(elty, :Species))
            if unique(elt.Species)[1] == "F_all" || unique(elt.Species)[1] == "N"
                plt.scatter(elt.Time, elt.Reading.*10^3, marker = "x", s = 50, linewidth = 4, color = colour_palette[idx], alpha = 0.5)  #F_all and N need to be scaled differently to C
            else
                plt.scatter(elt.Time, elt.Reading, marker = "x", s = 50, linewidth = 4, color = colour_palette[idx], alpha = 0.5)
            end
        end
        #
        # plot the ODE trajectories
        #
        Obs = simulate(Example_ODE, total_p_draws[i, 2:end], tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, cb = cb)
        Obs_low = simulate(Example_ODE, vcat(total_p_draws[i, 1], total_p_draws[i, 3:end]), tspan, solver, saveat; no_of_species = no_of_species, x0 = x0, cb = cb)
        if idy == 1   #if condition is High IC
            plt.plot(Obs.t, (Obs[1, :] .+ Obs[4, :]).*10^6, color = "darkgreen", linewidth = 3, alpha = 0.6)  #F_all
            plt.plot(Obs.t, Obs[2, :].*10^3, color = "grey", linewidth = 3, alpha = 0.6)  #C
            plt.plot(Obs.t, Obs[3, :].*10^6, color = "navy", linewidth = 3, alpha = 0.6)  #N
        else
            plt.plot(Obs_low.t, (Obs_low[1, :] .+ Obs_low[4, :]).*10^6, color = "darkgreen", linewidth = 3, linestyle = "--", alpha = 0.6)  #C
        end
        plt.xlabel("Time (hrs)", fontsize = 20)
        plt.tick_params(labelsize = 20)
        plt.yscale("log")
    end
end
tight_layout()
# savefig("p_sets.pdf", transparent = true)
#
# store parameter values
#
cd("data")
writedlm("parameters.csv", total_p_draws, ';')
