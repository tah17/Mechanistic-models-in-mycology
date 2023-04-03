#
# CREATES n_sims FAKE DATA SETS THAT MIMIC THE DATA SPARSITY OF THE PREVIOUSLY COLLECTED IN VIVO DATA IN MYCOLOGY
#
using Random
using DelimitedFiles
using Distributions
using DataFrames
using CSV
using Statistics
Random.seed!(96421658521890787)
include("../../src/ODEs.jl")
include("../../src/data-handling.jl")
n_sims = 100  #number of simulated reps of the simulation study
path = joinpath(@__DIR__, "../..")
#
# read in real data
#
real_data = CSV.read("/$path/data/data.csv", DataFrame)
#
# ODE arguments
#
solver = RK4()
x0 = missing  #we treat the ICs as parameters
t_pts = sort(unique(real_data.Time))
tspan = (0.0, t_pts[end])
saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
ode_species = ["F", "C", "N", "Fd"]
no_of_species = length(ode_species)
ODE = TotalDEInput(Example_ODE, tspan, solver, saveat, x0, no_of_species)
function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
  x[1] < 1e-6
end
function affect!(integrator)   #ODE argument to implement bounds in ODE solution
   integrator.u[1] = 0
end
cb = DiscreteCallback(condition, affect!)
maxiters = Int64(1e4)   #maximum iterations of ODE solver
p_ensemble = readdlm("/$path/data/parameters.csv", ';', Float64)   #get "true" parameters
no_of_param_set = size(p_ensemble)[1]
no_of_params= size(p_ensemble)[2]
noises = ["none", "low", "med", "high", "max"]
#
# user defined function that takes in the real data set (dataframe), the experimental condition e.g. "Low IC" and the ODE arguments needed to generate a fake data set.
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
        p = vcat(p_draw[1], p_draw[3:end])   #low IC for fungal burden is specified at p_draw[1]
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
cd("/$path/fake-data")
try
    mkdir("fake-data-1")
    cd("fake-data-1")
catch err
    cd("fake-data-1")
end
for idx in 1:no_of_param_set
    try
        mkdir("parameter-set-$idx")
        cd("parameter-set-$idx")
    catch err
        cd("parameter-set-$idx")
    end
    for (idy, noise) in enumerate(noises)
        try
            mkdir("noise-$idy")
            cd("noise-$idy")
        catch err
            cd("noise-$idy")
        end
        for idz in 1:n_sims
            dfs = [DataFrame() for elt in unique(real_data.Condition)]  #create different dataframes for the different experimental conditions
            for (idw, elt) in enumerate(unique(real_data.Condition))
                fake_species_data = user_defined(real_data, elt, ode_species, Example_ODE, p_ensemble[idx, :], solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
                if noise == "none"
                    noisy_data = fake_species_data
                else
                    noisy_data = add_noise(fake_species_data, noise, real_data, elt)
                end
                dfs[idw] = emulate_sparsity(noisy_data, real_data, elt)
            end
            df = join([df for df in dfs]..., kind  = :outer, on = intersect([names(df) for df in dfs]...))   #join the dataframes
            CSV.write("fake-data-set-$idz.csv", df)
        end
        cd("../")
    end
    cd("../")
end
