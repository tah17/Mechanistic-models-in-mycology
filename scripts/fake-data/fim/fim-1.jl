#
#  BATCH JOB SCRIPT THAT CALCULATES THE FISHER INFORMATION MATRIX (FIM) CORRESPONDING TO A SINGLE MLE, WHERE FIM CALCULATIONS FOR A SINGLE BATCH JOB ARE PARALLELISED USING nworkers() NUMBER OF WORKERS.
#  HERE FAKE DATA IS DATA THAT MIMICS THE DATA SPARSITY OF THE PREVIOUSLY COLLECTED DATA IN MYCOLOGY (fake-data-1.jl)
#  the script will be run as a batch job with 1-500 total jobs, with the first 1-100 jobs being n_sims fits for 0 noise, then 100-200 jobs for "low" noise, 200-300 for "med" noise, 300-400 for "high" noise and
#  finally 400-500th jobs for the maximum levels of noise. For each batch job i, the script will be parallelised using nworkers() - which split up the bootstrapping process.
#
using Distributed
# addprocs(5);  #local testing
@everywhere begin
    using DelimitedFiles
    using Optim
    using PyCall
    using Random
    using Future
    using CSV
    using DataFrames
end
include("../../../src/fit.jl")
include("../../../src/data-handling.jl")
include("../../../src/theoretical.jl")
@everywhere begin
    include(joinpath(@__DIR__, "../../../src/fit.jl"))
    include(joinpath(@__DIR__, "../../../src/data-handling.jl"))
    include(joinpath(@__DIR__, "../../../src/theoretical.jl"))
    @pyimport scipy.optimize as so
    n_sims = 100
    path = joinpath(@__DIR__, "../../..")
end
#
# julia RNGs
#
@everywhere Random.seed!(3744613754752089)
algo = ARGS[1]  #scipy algo
p_set = parse(Int64, ARGS[2])
chosen_noise = parse(Int64, ARGS[3])
@everywhere begin
    algo = $algo
    p_set = $p_set
    chosen_noise = $chosen_noise
end
#
# read in the fake data
#
all_data = Array{DataFrame,1}(undef, n_sims)
for i in collect(1:n_sims)
    all_data[i] = CSV.read("/$path/fake-data/fake-data-1/parameter-set-$p_set/noise-$chosen_noise/fake-data-set-$i.csv", DataFrame)
end
@everywhere begin
    #
    # ODE arguments
    #
    x0 = missing  #initial conditions
    solver = RK4()
    p_ensemble = readdlm("/$path/data/parameters.csv", ';', Float64)  #get "true" parameters
    p = p_ensemble[p_set, :]
    noise = false  #noise not inferred
    t_pts = sort(unique($all_data[1].Time))
    tspan = (0.0, t_pts[end])  #timespan
    saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
    ode_species = ["F", "C", "N", "Fd"]
    no_of_species = length(ode_species)
    number_of_exp = length(unique($all_data[1].Experiment))
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
    #  user defined function needed for calculation of the FIM. Takes an ODE solution & arguments needed to simulate the ODE solution
    #  and returns a dictionary where key are string of the measured species (e.g. "F_all") and value is a 2D array of size (time pts x no of exps)
    #
    #  NOTE - THIS USER DEFINED FUNCTION HAS A DIFFERENT ARGUMENT FORMAT TO ALL OTHER SCRIPTS (E.G. fit-data-i.jl AND bootstrap-i.jl)
    #
    function user_defined(real_data::DataFrame,
                          condition::String,
                          ode_species::Array{String,1},
                          sol::T where T<:DESolution,
                          p_draw::AbstractArray{T,1} where T<:Any,  #parameters will no longer be Floats when ODE is solved in Dual numbers
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
        sim_dat = get_fake_data(real_data, condition, ode_species, sol, p, solver, saveat;
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
end
#
# read in MLEs
#
working_result = Array{Float64,2}(undef, no_of_params + 1, n_sims)  #plus 1 for the SSE
for i in 1:n_sims
    working_result[:, i] = readdlm("/$path/data/fitted-data/fake-data/fitted-data-1/$algo/parameter-set-$p_set/noise-$chosen_noise/working-result-$i.csv", ';', Float64)
end
θ_hats, costs = working_result[1:end-1, :], working_result[end, :]
#
#  calculate FIMs
#
F = Array{Union{Float64, Missing},3}(undef, no_of_params, no_of_params, n_sims)
for i in 1:n_sims
    F[:, :, i] = FIM(θ_hats[:, i], user_defined, all_data[i], ODE, noise = noise)
end
cd("/$path/data")
try
    mkdir("fims-1")
    cd("fims-1")
catch err
    cd("fims-1")
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
for i in 1:n_sims
    elt = convert(Array{Any,2}, F[:, :, i])
    elt[ismissing.(elt)] .= "NA"
    writedlm("fims-$i.csv", elt, ';')
end
