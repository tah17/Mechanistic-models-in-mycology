#
#  BATCH JOB SCRIPT THAT CALCULATES BOOTSTRAP ESTIMATES (TO THEN BE CONVERTED TO CONFIDENCE INTERVALS), WHERE BOOTSTRAPPING FOR A SINGLE BATCH JOB IS PARALLELISED USING nworkers() NUMBER OF WORKERS.
#  HERE FAKE DATA IS DATA THAT MIMICS THE DATA SPARSITY OF THE CONVENTIONAL EXPERIMENTAL DESIGN (fake-data-2.jl)
#  the script will be run as a batch job with 1-500 total jobs, with the first 1-100 jobs being n_sims fits for 0 noise, then 100-200 jobs for low noise, 200-300 for "med" noise, 300-400 for "high" noise and
#  finally 400-500th jobs for the maximum levels of noise. For each batch job i, the script will be parallelised using nworkers() - which split up the bootstrapping process.
#
using Distributed
# addprocs(2);  #local testing
@everywhere begin
    using DelimitedFiles
    using Optim
    using PyCall
    using Random
    using Future
    using CSV
end
include("../../../src/data-handling.jl")
include("../../../src/fit.jl")
include("../../../src/bootstrap.jl")
@everywhere begin
    include(joinpath(@__DIR__, "../../../src/data-handling.jl"))
    include(joinpath(@__DIR__, "../../../src/fit.jl"))
    include(joinpath(@__DIR__, "../../../src/bootstrap.jl"))
    @pyimport scipy.optimize as so
    n_sims = 100
    b_reps = 100
    path = joinpath(@__DIR__, "../../..")
end
#
# julia RNGs needed for bootstrapping
#
@everywhere Random.seed!(3744613754752089)
tot_RNGs = Array{MersenneTwister,1}(undef, n_sims*nworkers())   #need a new stream for each simulated replicate (n_sims) but also for each worker with access to the script (nworkers())
RNG = MersenneTwister(3744613754752089)
tot_RNGs[1] = RNG
for i in 2:n_sims*nworkers()  #generate streams from seed
    RNG_jump = Future.randjump(tot_RNGs[i - 1], big(10)^20)  #default jump size
    tot_RNGs[i] = RNG_jump
end
batch_no = #read in batch number
batch_rg = mod(batch_no, n_sims)  #only need to have independent streams for the same noise i.e. for 100 batch jobs
if batch_rg == 0
    batch_rg = n_sims
end
batch_start = Int64(nworkers()*(batch_rg - 1) + 1)
batch_end = Int64(nworkers()*(batch_rg))
julia_rngs = tot_RNGs[batch_start:batch_end]   #pick the nworkers()' streams based off the batch number
#
# python RNGs (for SciPy algos)
#
@everywhere @pyimport numpy as np
@everywhere @pyimport numpy.random as npr
sg = npr.SeedSequence(921009879638187)
global bit_generator = npr.MT19937(sg)
tot_rg = Array{Any,1}(undef, n_sims*nworkers())
for i in 1:n_sims*nworkers()   #again, need a new stream for each simulated replicate (n_sims) but also for each worker with access to the script (nworkers())
    tot_rg[i] = npr.Generator(bit_generator)
    global bit_generator = bit_generator.jumped()
end
python_rngs = tot_rg[batch_start:batch_end]  #again, pick the nworkers() streams based off the batch number
#
# read in arguments passed to command line - details what parameter set to read in (used to generate the fake data in fake-data-2.jl)
#
algo = ARGS[1]   #scipy algo
p_set = parse(Int64, ARGS[2])
chosen_noise = ceil(Int64, batch_no/n_sims)
@everywhere begin   #make arguments known to all workers
    algo = $algo
    p_set = $p_set
    chosen_noise = $chosen_noise
end
#
# records job IDs - NOTE need file bootstrap-2-ids.txt already created and in same directory as submission file
#
job_id = #get job id
job_recorded = false
open("bootstrap-2-ids.txt") do io
  for ln in eachline(io)
    if occursin(job_id, ln)
       global job_recorded = true
    end
  end
end;
open("bootstrap-2-ids.txt", "a") do io
  if !job_recorded
      write(io, "Algo $algo with parameter $p_set: $job_id\n")
  end
end;
#
# read in the fake data set that was used for fitting (created by fake-data-2.jl) specific to this batch no.
#
fake_data = CSV.read("/$path/fake-data/fake-data-2/parameter-set-$p_set/noise-$chosen_noise/fake-data-set-$batch_rg.csv", DataFrame)
@everywhere begin
    #
    # ODE arguments
    #
    ode_species = ["F", "C", "N", "Fd"]
    no_of_species = length(ode_species)
    x0 = missing
    solver = RK4()
    p_ensemble = readdlm("/$path/data/parameters.csv", ';', Float64)  #get "true" parameters
    p = vcat(p_ensemble[p_set, 1], p_ensemble[p_set, 3:end])
    noise = false  #noise not inferred
    t_pts = sort(unique($fake_data.Time))
    tspan = (0.0, t_pts[end])  #timespan
    saveat = minimum([t_pts[i+1] - t_pts[i] for i in 1:length(t_pts) - 1])
    number_of_exp = length(unique($fake_data.Experiment))
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
end
#
# user defined function that takes in the fake data set (dataframe), the experimental condition e.g. "Low IC" and the ODE arguments needed to generate a fake data set
# and outputs a dictionary that has values of 2D arrays of data (time pts x no of exps) and keys that are the measured species name (String).
#
@everywhere function user_defined(real_data::DataFrame,   #all workers require access to this fn
                                  condition::String,
                                  ode_species::Array{String,1},
                                  Example_ODE::Function,
                                  p_draw::AbstractArray{Float64,1},
                                  solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                                  saveat::Real;
                                  x0::Union{AbstractArray{Float64,1}, Missing} = nothing,
                                  maxiters::Union{Int64, Nothing} = nothing,
                                  cb::Union{Nothing, Any} = nothing)

    measured_species = ["F_all", "C", "N"]
    sim_dat = get_fake_data(real_data, condition, ode_species, Example_ODE, p_draw, solver, saveat;
                            x0 = x0, maxiters = maxiters, cb = cb)
    f_all = sim_dat[:, :, 1] + sim_dat[:, :, 4]  #Fall = F + Fd
    c = sim_dat[:, :, 2]
    return Dict(measured_species[1] => f_all, measured_species[2] => c)
end
#
# read in MLEs from fitting to the fake data (generated by running fit-data-2.jl) specific to this batch no.
#
working_result = readdlm("/$path/data/fitted-data/fake-data/fitted-data-2/$algo/parameter-set-$p_set/noise-$chosen_noise/working-result-$(batch_rg).csv", ';', Float64)
θ_hats, costs = working_result[1:end-1, :], working_result[end, :]
θ_hat = θ_hats[:, 1]   #note that there is no F(high)_0 initial condition parameter
#
# fit the ODE
#
if algo == "BH"
    @everywhere begin
        pyfun = so.basinhopping
        p0s_tot = [vcat( (10 .^ vcat(rand($RNG, Normal(4, 1)), rand($RNG, Normal(0, 1), no_of_species - 1)))./10^6, rand($RNG, Truncated(Normal(0, 1), 0, 4), no_of_params - no_of_species)) for i in 1:n_sims]
        p0 = p0s_tot[$batch_rg]  #pick the IC based off the batch number - note this p0 is not used for the actual fitting
        niter = 1000  #default
        pyoptim = PyOptimiser(pyfun, p0, niter)
        function wrap_pyfun(cost::Function, p0::Array{Float64,1}; niter::Union{Int64, Nothing}, seed::Any) #wrapper avoids an error encountered with some PyCall functions and @distributed macro
            return pyoptim.pyfun(cost, p0, niter = niter, seed = seed)
        end
    end
    b_θs = SharedArray{Float64,2}(length(pyoptim.p0), b_reps)
    @sync @distributed for i in 1:b_reps  #perform the bootstrap fitting - needs @sync to avoid lack of reproducibility that happens with dynamic dispatch
        p0_i = vcat( (10 .^ vcat(rand(julia_rngs[myid() - 1], Normal(4, 1)), rand(julia_rngs[myid() - 1], Normal(0, 1), no_of_species - 1)))./10^6, rand(julia_rngs[myid() - 1], Truncated(Normal(0, 1), 0, 4), no_of_params - no_of_species)) #starting guess for algo that depends on the a specified stream
        data_b = bootstrap_sample(fake_data, julia_rngs[myid() - 1])  #sample a bootstap data set by bootstrapping by experiments, and use julia seed labelled by worker ID (which begin at 2 and end at nworkers() + 1)
        res = wrap_pyfun(x -> residuals(ODE.DE, user_defined, data_b, ode_species, x, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb), convert(Array{Float64,1}, p0_i), niter = pyoptim.niter, seed = python_rngs[myid() - 1])  #pick a stream based off worker ID (which begin at 2 and end at nworkers() + 1)
        b_θs[:, i] = res["x"]
    end
elseif algo == "DE"
    @everywhere begin
        pyfun = so.differential_evolution
        p_range = vcat(tuple(0, 1), [tuple(0, 4) for i in 1:(no_of_params - 1)])  #parameter ranges needed for DE algo
        pyoptim = PyOptimiser(pyfun, p_range)
        function wrap_pyfun(cost::Function, p_range::AbstractArray{Tuple{T,T},1} where T <: Real; seed::Any) #wrapper avoids an error encountered with some PyCall functions and @distributed macro
            return pyoptim.pyfun(cost, p_range, seed = seed)
        end
    end
    b_θs = SharedArray{Float64,2}(length(pyoptim.p_range), b_reps)
    @sync @distributed for i in 1:b_reps  #perform the bootstrap fitting - needs @sync to avoid lack of reproducibility that happens with dynamic dispatch
        data_b = bootstrap_sample(fake_data, julia_rngs[myid() - 1])  #sample a bootstap data set by bootstrapping by experiments, and use julia seed labelled by worker ID (which begin at 2 and end at nworkers() + 1)
        res = wrap_pyfun(x -> residuals(ODE.DE, user_defined, data_b, ode_species, x, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb), pyoptim.p_range, seed = python_rngs[myid() - 1])  #pick a stream based off worker ID (which begin at 2 and end at nworkers() + 1)
        b_θs[:, i] = res["x"]
    end
else
    @everywhere begin
        pyfun = so.least_squares
        p0s_tot = [vcat( (10 .^ vcat(rand($RNG, Normal(4, 1)), rand($RNG, Normal(0, 1), no_of_species - 1)))./10^6, rand($RNG, Truncated(Normal(0, 1), 0, 4), no_of_params - no_of_species)) for i in 1:n_sims]
        p0 = p0s_tot[$batch_rg]   #not used - just placeholder for PyOptimiser struct
        pyoptim = PyOptimiser(pyfun, p0)
        function wrap_pyfun(cost::Function, p0::Array{Float64,1}) #wrapper avoids an error encountered with some PyCall functions and @distributed macro
            try
                return pyoptim.pyfun(cost, p0)
            catch  #if error thrown return the initial parameters and the worst possible cost value
                res = Dict()
                res["x"] = p0
                count = 0
                for (idx, elt) in enumerate(unique(fake_data.Condition))
                    tmp_df = filter(x -> x.Condition == elt, fake_data)
                    df = by(tmp_df, :Species) do df
                        DataFrame(max = maximum(df[:Reading]))
                    end
                    count += sum(df.max .^ 2)
                end
                res["fun"] = 500*count  #1000 times the "worst result" - (taking the max values of each data point)
                return res
            end
        end
    end
    b_θs = SharedArray{Float64,2}(length(pyoptim.p0), b_reps)  #perform the bootstrap fitting - needs @sync to avoid lack of reproducibility that happens with dynamic dispatch
    @sync @distributed for i in 1:b_reps
        p0_i = vcat((10 .^ vcat(rand(julia_rngs[myid() - 1], Normal(4, 1)), rand(julia_rngs[myid() - 1], Normal(0, 1), no_of_species - 1)))./10^6, rand(julia_rngs[myid() - 1], Truncated(Normal(0, 1), 0, 4), no_of_params - no_of_species))  #starting guess for algo that depends on the a specified stream
        data_b = bootstrap_sample(fake_data, julia_rngs[myid() - 1])  #sample a bootstap data set by bootstrapping by experiments, and use julia seed labelled by worker ID (which begin at 2 and end at nworkers() + 1)
        res = wrap_pyfun(x -> residuals(ODE.DE, user_defined, data_b, ode_species, x, tspan, solver, saveat; noise = noise, maxiters = maxiters, cb = cb), convert(Array{Float64,1}, p0_i))
        b_θs[:, i] = res["x"]
    end
end
cd("/$path/data")
try
    mkdir("bootstrap-2")
    cd("bootstrap-2")
catch err
    cd("bootstrap-2")
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
writedlm("bootstrap-estimates-$(batch_rg).csv", b_θs, ';')
