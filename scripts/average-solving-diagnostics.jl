#
# SCRIPT TO CHECK THE ODE SOLVING DIAGNOSTICS TO MOTIVATE USING A SMALLER MAXITERS THAN THE DEFAULT
#
using DifferentialEquations
using DelimitedFiles
using PyPlot; ENV["MPLBACKEND"]="tkagg"; pygui(true);
using Random
using CSV
using DataFrames
Random.seed!(3744613754752089)
include("../src/ODEs.jl")
include("../src/functions.jl")
filepath = joinpath(@__DIR__, "..")

x0 = missing  #infer i.c.s
solver = RK4()
real_data = CSV.read("$filepath/data/data.csv", DataFrame)
noise = false  #don't estimate the noise
t_pts = sort(unique(real_data.Time))
tspan = (0.0, t_pts[end])
saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
number_of_exp = length(unique(real_data.Experiment))
ode_species = ["F", "C", "N", "Fd"]
no_of_species = length(ode_species)
ODE = TotalDEInput(Example_ODE, tspan, solver, saveat, no_of_species)
function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
  x[1] < 1e-6
end
function affect!(integrator)
   integrator.u[1] = 0
end
cb = DiscreteCallback(condition, affect!)
maxiters = Int64(1e6)  #default
N = Int64(1e5)  #number of parameters sampled
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
                    Truncated(Normal(0.09, 0.045), 0, 2)], N)
p_draw = hcat(p_draw...)
#
# avergae solving time
#
start1 = time()
sims = (x -> simulate(Example_ODE, vcat([10^p_draw[x, i]/10^6 for i in 1:5], p_draw[x, 6:end]), tspan, solver, saveat; no_of_species = no_of_species, noise = noise, maxiters = maxiters, cb = cb)).(collect(1:N))
time_elpd_max = time() - start1
time_elpd_max/N
#
# average ODE iterations
#
sim_iters = Array{Int64, 1}(undef, N)
for (idx, elt) in enumerate(sims)
    try
        sim_iters[idx] = elt.destats.naccept + elt.destats.nreject
    catch err
        sim_iters[idx] = Int64(maxiters)
    end
end
#
# summary statistics of solving iterations
#
median(sim_iters)
var(sim_iters)
mean(sim_iters)
sqrt(var(sim_iters))/sqrt(Int64(1e4))
quantile(sim_iters, 0.95)
sims[findmax(sim_iters)[2]]
plt.figure(figsize=(5,5))
plt.hist(sim_iters, bins=5000, color = "lightgrey", ec="black", lw=3)
plt.axvline(median(sim_iters), color = "#4daf4a", linestyle="dashed", linewidth=2)
plt.axvline(mean(sim_iters), color = "#377eb8", linestyle="dashed", linewidth=2)
plt.axvline(1e4, color = "#e41a1c", linestyle="dashed", linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.tick_params(labelsize = 20)
plt.xlabel("Max Iters.", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
tight_layout()
savefig("solving_iters.pdf", transparent = true)
tight_layout()
maxiters = Int64(floor(quantile(sim_iters, 0.80)))
maxiters = Int64(floor(median(sim_iters)))
tmp = convert(Vector{Any}, sims)
tmp[ismissing.(tmp)] .= "NA"
# if the max is hit a lot look at the iteraetions of ODE solving without the maximum
median(sim_iters[sim_iters .!= 1e6])
var(sim_iters[sim_iters .!= 1e6])
mean(sim_iters[sim_iters .!= 1e6])
quantile(sim_iters[sim_iters .!= 1e6], 0.95)
plt.hist(sim_iters[sim_iters .!= 1e6])
plt.hist(sim_iters)
