#
# BATCH JOB SCRIPT THAT CALCULATES THE PROFILE LIKLEIHOOD CONFIDENCE INTERVALS FROM FAKE DATA
# HERE FAKE DATA IS DATA THAT MIMICS THE DATA SPARSITY OF THE PREVIOUSLY COLLECTED DATA IN MYCOLOGY (fake-data-1.jl)
#  the script will be run as a batch job with 1-500 total jobs, with the first 1-100 jobs being n_sims fits for 0 noise, then 100-200 jobs for low noise, 200-300 for "med" noise, 300-400 for "high" noise and
#  finally 400-500th jobs for the maximum levels of noise.
#
using DelimitedFiles
using Optim
using PyCall
using Random
using Future
using LikelihoodProfiler
using CSV
include("../../../src/data-handling.jl")
include("../../../src/bootstrap.jl")
@pyimport scipy.optimize as so
n_sims = 100
path = joinpath(@__DIR__, "../../..")
Random.seed!(3744613754752089)  #julia RNG
batch_no =  #read in batch number
batch_rg = mod(batch_no, n_sims)   #only need to have independent streams for the same noise i.e. for 100 batch jobs
if batch_rg == 0
    batch_rg = n_sims
end
#
# read in arguments passed to command line - details what parameter set to read in (used to generate the fake data in fake-data-1.jl)
#
algo = ARGS[1]   #scipy algo
p_set = parse(Int64, ARGS[2])
chosen_noise = ceil(Int64, batch_no/n_sims)
#
# records job IDs - NOTE need file profile-1-ids.txt already created and in same directory as submission file
#
job_id = #get job id
job_recorded = false
open("profile-1-ids.txt") do io
  for ln in eachline(io)
    if occursin(job_id, ln)
       global job_recorded = true
    end
  end
end;
open("profile-1-ids.txt", "a") do io
  if !job_recorded
      write(io, "Algo $algo with parameter $p_set: $job_id\n")
  end
end;
#
# read in the fake data set that was used for fitting (created by fake-data-1.jl) specific to this batch no.
#
fake_data = CSV.read("/$path/fake-data/fake-data-1/parameter-set-$p_set/noise-$chosen_noise/fake-data-set-$batch_rg.csv", DataFrame)
#
# ODE arguments
#
ode_species = ["F", "C", "N", "Fd"]
no_of_species = length(ode_species)
x0 = missing
solver = RK4()
p_ensemble = readdlm("/$path/data/parameters.csv", ';', Float64)  #get "true" parameters
p = p_ensemble[p_set, :]
noise = false  #noise not inferred
t_pts = sort(unique(fake_data.Time))
tspan = (0.0, t_pts[end])  #timespan
saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
number_of_exp = length(unique(fake_data.Experiment))
no_of_params = length(p)
ODE = TotalDEInput(Example_ODE, tspan, solver, saveat, x0, no_of_species)
order_params = floor.(log10.(abs.(p)))
function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
    x[1] < 1e-6
end
function affect!(integrator)
    integrator.u[1] = 0
end
cb = DiscreteCallback(condition, affect!)
maxiters = Int64(1e4)
#
# user defined function that takes in a fake data set (dataframe), the experimental condition e.g. "High IC" and the ODE arguments needed to generate a fake data set
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
# read in MLEs from fitting to the fake data (generated by running fit-data-1.jl) specific to this batch no.
#
working_result = readdlm("/$path/data/fitted-data/fake-data/fitted-data-1/$algo/parameter-set-$p_set/noise-$chosen_noise/working-result-$(batch_rg).csv", ';', Float64)
θ_hats, costs = working_result[1:end-1, :], working_result[end, :]
θ_hat = θ_hats[:, 1]
alphas = collect(0.01:0.05:0.99)  #confidence interval alpha levels
#
# calculate profile CIs
#
scan_bounds = vcat(tuple(1e-9, 1.0 - 1e-9), tuple(1e-9, 1000.0 - 1e-9), [tuple(1e-9, 4.0 - 1e-9) for i in 1:(no_of_params - 2)])
theta_bounds = vcat(tuple(0.0, 1.0), tuple(0.0, 1000.0), [tuple(0.0, 4.0) for i in 1:(no_of_params - 2)])
ci = Array{Tuple{Float64,Float64},2}(undef, size(θ_hats)[1], length(alphas))
for (idx, alpha) in enumerate(alphas)
    l = Array{Any,1}(undef, length(θ_hats))
    u = Array{Any,1}(undef, length(θ_hats))
    confit_intervals = Vector{ParamInterval}(undef, length(θ_hat))
    for i in eachindex(θ_hat)
        confit_intervals[i] = get_interval(θ_hat,   #calls get_interval() from LikelihoodProfiler
                                           i,
                                           (p) -> residuals(ODE.DE, user_defined, fake_data, ode_species, p, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb),
                                           :CICO_ONE_PASS,
                                           loss_crit = residuals(ODE.DE, user_defined, fake_data, ode_species, θ_hat, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb) .+ (quantile.(Chisq(1), alpha)./2),
                                           theta_bounds = theta_bounds,
                                           scan_bounds = scan_bounds[i],
                                           # local_alg = :LN_COBYLA,
                                           scale = fill(:log, length(θ_hat))
                                           )
        l[i], u[i] = confit_intervals[i].result[1].value, confit_intervals[i].result[2].value
        if isnothing(l[i])  #if CI bounds are not located, then return the max theta bounds defined
            l[i] = theta_bounds[i][1]
        end
        if isnothing(u[i])
            u[i] = theta_bounds[i][2]
        end
    end
    ci[:, idx] = tuple.(l, u)
end
cd("/$path/data")
try
    mkdir("profile-intervals-1")
    cd("profile-intervals-1")
catch err
    cd("profile-intervals-1")
end
try
    mkdir("$algo")
    cd("$algo")
catch err
    cd("$algo")
end
try
    mkdir("parameter-set-$p_set")
    cd("parameter-set-$p_set")
catch err
    cd("parameter-set-$p_set")
end
try
    mkdir("noise-$chosen_noise")
    cd("noise-$chosen_noise")
catch err
    cd("noise-$chosen_noise")
end
println("saving results")
writedlm("profile-intervals-$(batch_rg).csv", ci, ';')
