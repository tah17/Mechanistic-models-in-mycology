#
#  BATCH JOB SCRIPT THAT FITS THE MODEL TO ITS OWN FAKE DATA - HERE FAKE DATA IS DATA THAT MIMICS THE DATA SPARSITY OF THE PREVIOUSLY COLLECTED DATA IN MYCOLOGY
#  AND A PROPOSED CONVENTIONAL EXPERIMENTAL DESIGN BUT THE EXPERIMENT IS DONE IN IMMUNOCOMPETENT MICE AND NEUTROPHILS (N) ARE RECORDED
#  AND THE VARIABLES ARE RECORDED AT TIMES 12, 32 AND 72 HOURS VS 24, 48 AND 72 HRS (fake-data-5.jl)
#  the script will be run as a btach job with 1-500 total jobs, with the first 1-100 jobs being n_sims fits for 0 noise, then 100-200 jobs for low noise, 200-300 for "med" noise, 300-400 for "high" noise and
#  finally 400-500th jobs for the maximum levels of noise.
#
using DelimitedFiles
using Optim
using PyCall
using Random
using Future
using SharedArrays
using DataFrames
using CSV
n_sims = 100
path = joinpath(@__DIR__, "../../..")
#
# python RNGs to be fed into SciPy algos
#
Random.seed!(3744613754752089)
RNG = MersenneTwister(3744613754752089)
@pyimport numpy as np
@pyimport numpy.random as npr
sg = npr.SeedSequence(921009879638187)
global bit_generator = npr.MT19937(sg)
tot_rg = Array{Any,1}(undef, n_sims)
for i in 1:n_sims  #generate n_sims streams from one seed
    tot_rg[i] = npr.Generator(bit_generator)
    global bit_generator = bit_generator.jumped()
end
batch_no = #get batch no. of job
batch_rg = mod(batch_no, n_sims)   #only need to have independent streams for the same noise i.e. for 100 batch jobs
if batch_rg == 0
    batch_rg = n_sims
end
rg = tot_rg[batch_rg]
include("../../../src/data-handling.jl")
include("../../../src/fit.jl")
@pyimport scipy.optimize as so
#
# read in arguments passed to command line - details what parameter set to read in (used to generate the fake data in fake-data.jl)
#
algo = ARGS[1]  #scipy algorithm
p_set = parse(Int64, ARGS[2])
chosen_noise = ceil(Int64, batch_no/n_sims)
#
# records job IDs - NOTE need file fit-5-ids.txt already created and in same directory as sub file
#
job_id = #get job id
job_recorded = false
open("fit-5-ids.txt") do io
  for ln in eachline(io)
    if occursin(job_id, ln)
       global job_recorded = true
    end
  end
end;
open("fit-5-ids.txt", "a") do io
  if !job_recorded
      write(io, "Algo $algo with parameter $p_set: $job_id\n")
  end
end;
#
# read in the fake data set
#
fake_data = CSV.read("/$path/fake-data/fake-data-5/parameter-set-$p_set/noise-$chosen_noise/fake-data-set-$batch_rg.csv", DataFrame)
#
# ODE arguments
#
ode_species = ["F", "C", "N", "Fd"]
no_of_species = length(ode_species)
x0 = missing  #infer i.c.s
solver = RK4()
p_ensemble = readdlm("/$path/data/parameters.csv", ';', Float64)  #get "true" parameters
p = p_ensemble[p_set, :]
noise = false  #noise not inferred
t_pts = sort(unique(fake_data.Time))
tspan = (0.0, t_pts[end])  #timespan
saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
number_of_exp = length(unique(fake_data.Experiment))
no_of_params = length(p)
ODE = TotalDEInput(Example_ODE, tspan, solver, saveat, no_of_species)
function condition(x, t, integrator)  # if fungal burden < 1 CEs then set F = 0
  x[1] < 1e-6
end
function affect!(integrator)
   integrator.u[1] = 0
end
cb = DiscreteCallback(condition, affect!)
maxiters = Int64(1e4)
order_params = floor.(log10.(abs.(p)))
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
        return Dict(measured_species[1] => f_all, measured_species[2] => c)
    else
        throw(ArgumentError("$condition is not a valid experimental condition"))
    end
end
#
# set up fitting specifications and fits ODE
#
if algo == "BH"
    pyfun = so.basinhopping
    p0s_tot = [vcat((10 .^ vcat(rand(RNG, Normal(4, 1)), rand(RNG, Normal(7, 1)), rand(RNG, Normal(0, 1), no_of_species - 1)))./10^6, rand(RNG, Truncated(Normal(0, 1), 0, 4), no_of_params - no_of_species - 1)) for i in 1:n_sims]  #generates 100 ICs for the fitting algo
    p0 = p0s_tot[batch_rg]  #pick the IC based off the batch number
    niter = 1000
    pyoptim = PyOptimiser(pyfun, p0, niter)
    function wrap_pyfun(cost::Function, p0::Array{Float64,1}; niter::Union{Int64, Nothing}, seed::Any) #wrapper avoids an error encountered with some PyCall functions and @distributed macro
        return pyoptim.pyfun(cost, p0, niter = niter, seed = seed)
    end
    #
    # fit the ODE
    #
    res = wrap_pyfun(x -> residuals(ODE.DE, user_defined, fake_data, ode_species, x, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb), convert(Array{Float64,1}, p0), niter = pyoptim.niter, seed = rg)
elseif algo == "DE"
    pyfun = so.differential_evolution
    p_range = vcat(tuple(0, 1), tuple(0, 1000), [tuple(0, 4) for i in 1:(no_of_params - 2)])
    pyoptim = PyOptimiser(pyfun, p_range)
    function wrap_pyfun(cost::Function, p_range::AbstractArray{Tuple{T,T},1} where T <: Real; seed::Any) #wrapper avoids an error encountered with some PyCall functions and @distributed macro
        return pyoptim.pyfun(cost, p_range, seed = seed)
    end
    #
    # fit the ODE
    #
    res = wrap_pyfun(x -> residuals(ODE.DE, user_defined, fake_data, ode_species, x, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb), pyoptim.p_range, seed = rg)
else  #least squares
    p0s_tot = [vcat((10 .^ vcat(rand(RNG, Normal(4, 1)), rand(RNG, Normal(7, 1)), rand(RNG, Normal(0, 1), no_of_species - 1)))./10^6, rand(RNG, Truncated(Normal(0, 1), 0, 4), no_of_params - no_of_species - 1)) for i in 1:n_sims]  #generates 100 ICs for the fitting algo
    pyfun = so.least_squares
    p0 = p0s_tot[batch_rg]  #pick the IC based off the batch number
    pyoptim = PyOptimiser(pyfun, p0)
    function wrap_pyfun(cost::Function, p0::Array{Float64,1}) #wrapper avoids an error encountered with some PyCall functions and @distributed macro
        try
            return pyoptim.pyfun(cost, p0)
        catch   #if error thrown return the initial parameters and the worst possible cost value
            res = Dict()
            count = 0
            for (idx, elt) in enumerate(unique(fake_data.Condition))
                tmp_df = filter(x -> x.Condition == elt, fake_data)
                df = by(tmp_df, :Species) do df
                    DataFrame(max = maximum(df[:Reading]))
                end
                count += sum(df.max .^ 2)
            end
            res["fun"] = 500*count  #1000 times the "worst result" - (taking the max values of each data point)
            res["x"] = p0
            return res
        end
    end
    #
    # fit the ODE
    #
    res = wrap_pyfun(x -> residuals(ODE.DE, user_defined, fake_data, ode_species, x, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb), convert(Array{Float64,1}, p0))
end
working_result = vcat(res["x"], res["fun"])
cd("/$path/data/fitted-data/fake-data")
try
    mkdir("fitted-data-5")
    cd("fitted-data-5")
catch err
    cd("fitted-data-5")
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
writedlm("working-result-$(batch_rg).csv", working_result, ';')
