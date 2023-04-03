#
# CREATES n_sims FAKE DATA SETS THAT MIMIC THE DATA SPARSITY OF THE PROPOSED CONVENTIONAL EXPERIMENTAL DESIGN
# BUT WITH REFINEMENTS OF: THE EXPERIMENT IS DONE IN IMMUNOCOMPETENT MICE & NEUTROPHILS ARE RECORDED
# THIS MEANS THIS FAKE DATA IS OF ALL THE PREVIOUSLY COLLECTED IMMUNOCOMPETENT IN VIVO DATA AND THE PROPOSED CONVENTIONAL EXPERIMENTAL DESIGN
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
no_of_exps = 10  #number of biological replicates
path = joinpath(@__DIR__, "../..")
#
# read in real data
#
real_data = CSV.read("/$path/data/data.csv", DataFrame)
#
# ODE arguments
#
solver = RK4()
x0 = missing
t_pts_quant_review = sort(unique(real_data.Time))  #time points of previous experimental data
tspan_quant_review = (0.0, t_pts_quant_review[end])
saveat_quant_review = minimum([t_pts_quant_review[i+1] - t_pts_quant_review[i] for i in 1:length(t_pts_quant_review) - 1])
t_pts = [24.0, 48.0, 72.0]  #time points of planned experimental data
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
p_ensemble = readdlm("/$path/data/parameters.csv", ';', Float64)  #get "true" parameters
no_of_param_set = size(p_ensemble)[1]
no_of_params = size(p_ensemble)[2]
noises = ["none", "low", "med", "high", "max"]
#
# user defined function (`quant_rev`) that takes in the real data set (dataframe), the experimental condition e.g. "High IC" and the ODE arguments needed to generate a fake data set
# and outputs a dictionary that has values of 2D arrays of data (time pts x no of exps) and keys that are the measured species name (String).
# This function is then passed to all downstream data handling functions so it's important to keep the input and output structure the same.
#
function quant_rev(real_data::DataFrame,
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
# user defined function `new_exp` that takes in the ODE arguments and time points needed to generate the fake data set
# and outputs a dictionary that has values of 2D arrays of data (time pts x no of exps) and keys that are the measured species name (String).
# This function is then passed to all downstream data handling functions so it's important to keep the input and output structure the same.
#
function new_exp(ode_species::Array{String,1},
                 Example_ODE::Function,
                 p_draw::AbstractArray{Float64,1},
                 t_pts::AbstractArray{T,1} where T<:Real,
                 solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                 saveat::Real,
                 no_of_exps::Int64;
                 x0::Union{AbstractArray{Float64,1}, Missing} = nothing,
                 maxiters::Union{Int64, Nothing} = nothing,
                 cb::Union{Nothing, Any} = nothing)

    measured_species = ["F_all", "C", "N"]
    p = vcat(p_draw[1], p_draw[3:end])
    sim_dat = get_fake_data(Example_ODE, ode_species, p, saveat, t_pts, solver, x0, no_of_exps;
                            maxiters = maxiters, cb = cb)
    f_all = sim_dat[:, 1:convert(Int64, no_of_exps/2), 1] + sim_dat[:, 1:convert(Int64, no_of_exps/2), 4]  #Fall = F + Fd and only 5 reps for F (as opposed to 10)
    c = sim_dat[:, :, 2]
    n = sim_dat[:, 1:convert(Int64, no_of_exps/2), 3]
    return Dict(measured_species[1] => f_all, measured_species[2] => c, measured_species[3] => n)
end
#
# new exp noises
#
data = filter(x -> !ismissing(x.Reading), real_data)
vars = []
maxs = []
for (idx, elt) in enumerate(unique(real_data.Condition))
    tmp_df = filter(x -> x.Condition == elt, data)
    df = by(tmp_df, :Species) do df
        DataFrame(m = mean(df[:Reading]), s² = var(df[:Reading]), max = maximum(df[:Reading]))
    end
    push!(vars, df.s²)
    push!(maxs, df.max)
end
new_exp_var = vcat(vars[2], vars[1][2:end])  #stores the vars of F_all (low IC), C and N (high IC)
new_exp_max = vcat(maxs[2], maxs[1][2:end])
cd("/$path/fake-data")
try
    mkdir("fake-data-4")
    cd("fake-data-4")
catch err
    cd("fake-data-4")
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
            dfs = [DataFrame() for elt in 1:3]
            for (idw, elt) in enumerate(unique(real_data.Condition))  #begin with making fake data sets of the previously collected in vivo data (quant review data)
                fake_species_data = quant_rev(real_data, elt, ode_species, Example_ODE, p_ensemble[idx, :], solver, saveat_quant_review; x0 = x0, maxiters = maxiters, cb = cb)
                if noise == "none"
                    noisy_data = fake_species_data
                else
                    noisy_data = add_noise(fake_species_data, noise, real_data, elt)
                end
                dfs[idw] = emulate_sparsity(noisy_data, real_data, elt)
            end  #then create fake data of potential experiment
            fake_new_exp = new_exp(ode_species, Example_ODE, p_ensemble[idx, :], t_pts, solver, saveat, no_of_exps; x0 = x0, maxiters = maxiters, cb = cb)
            if noise == "none"
                noisy_data = fake_new_exp
            else
                noisy_data = add_noise(fake_new_exp, noise, new_exp_var, data_maxs = new_exp_max)
            end
            starting_exp_no = maximum(vcat([parse.(Int, (x -> x[2]).(split.(unique(dfs[i].Experiment), "E"))) for i in 1:2]...))
            dfs[end] = make_df(noisy_data, t_pts, "Low IC", starting_exp = starting_exp_no)
            df = join([df for df in dfs]..., kind  = :outer, on = intersect([names(df) for df in dfs]...))
            CSV.write("fake-data-set-$idz.csv", df)
        end
        cd("../")
    end
    cd("../")
end
