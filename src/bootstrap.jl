using PyCall
using DifferentialEquations
using Distributions
using Distributed
using Random
using SharedArrays
using DataFrames
#
include("coverage.jl")

"""
    bootstrap_sample(dat::DataFrame, rng_j::MersenneTwister)

Bootstrap the data (`dat`) by resampling the experiments. Returns a dataframe of a new bootstrap data set.
"""

function bootstrap_sample(dat::DataFrame, rng::MersenneTwister)

  experiments = unique(dat.Experiment)
  experiment_idxs = collect(1:length(experiments))
  dfs = [DataFrame() for x in experiment_idxs]
  new_experiment_idxs = rand(rng, experiments, length(experiments))
  new_experiment_labels = ["B$i" for i in experiment_idxs]
  for i in experiment_idxs
    dfs[i] = filter(x -> x.Experiment == new_experiment_idxs[i], dat)
    dfs[i].Experiment = repeat([new_experiment_labels[i]], inner = length(dfs[i].Time))
  end
  return join([df for df in dfs]..., kind  = :outer, on = intersect([names(df) for df in dfs]...))
end

"""
    bootstrap_intervals(theta_hats_bstrap::AbstractArray{Float64,2}, alpha::Float64; <keyword arguments>)

Calculate the `alpha`-level bootsrap confidence intervals for previously calculated bootstrap parameter estimates `theta_hats_bstrap`  (size: number of parameters by bootstrap reps).
Will return the `type` of specified by bootsrap confidence intervals e.g. percentile, z-interval or a t-interval (note not proper bootstrap-t but just a bootstrap with t distribution).
If needed for interval calculation pass `theta_hat`; the 'best fit' parameter estimate from the original (not bootstrapped) data set.

"""
function bootstrap_intervals(theta_hats_bstrap::AbstractArray{Float64,2},  #add extended CI
                             alpha::Float64;
                             type::String = "percentile",
                             theta_hat::Union{AbstractArray{Float64,1}, AbstractArray{Int64,1}, Nothing} = nothing)  #confidence intervals for just one simulation rep to add functionality

  if !ininterval(alpha, (0, 1))
    throw(ArgumentError("The confidence level $alpha needs to be between 0 and 1."))
  end
  if (type == "z-interval" || type == "t-interval" || type == "corrected-t-interval") && isnothing(theta_hat)
    throw(ArgumentError("The $type confidence interval is centered around a statistic `theta_hat` to be provided"))
  else
    no_of_params, b_reps = size(theta_hats_bstrap)
    if type == "percentile"
      quantiles = [quantile(sort(theta_hats_bstrap[i, :]), [alpha/2, 1 - alpha/2]) for i in 1:no_of_params]  #alpha% quantiles of boostrap replicates for each parameter
      return tuple.((x -> x[1]).(quantiles), (x -> x[2]).(quantiles))  #return array of tuples - confidence intervals for each parameter
    elseif type == "z-interval"
      if no_of_params != length(theta_hat)
        throw(ArgumentError("You provided bootstrap estimates for $no_of_params parameters but $(length(theta_hat)) parameter estimates from the non-bootstrapped data set"))
      end
      quantiles = sqrt.(var(theta_hats_bstrap, dims = 2)) .* quantile(Normal(), alpha/2)  #normal dist interval
      return tuple.(theta_hat .- abs.(reshape(collect(quantiles), no_of_params)),  theta_hat .+ abs.(reshape(collect(quantiles), no_of_params)))
    elseif type == "t-interval"
      if no_of_params != length(theta_hat)
        throw(ArgumentError("You provided bootstrap estimates for $no_of_params parameters but $(length(theta_hat)) parameter estimates from the non-bootstrapped data set"))
      end
      quantiles = sqrt.(var(theta_hats_bstrap, dims = 2)) .* quantile(TDist(b_reps - 1), 1 - alpha/2)  #t-dist interval - better for small n
      return tuple.(theta_hat .- reshape(collect(quantiles), no_of_params),  theta_hat .+ reshape(collect(quantiles), no_of_params))
    elseif type == "corrected-t-interval"
      if no_of_params != length(theta_hat)
        throw(ArgumentError("You provided bootstrap estimates for $no_of_params parameters but $(length(theta_hat)) parameter estimates from the non-bootstrapped data set"))
      end
      quantiles = sqrt.(var(theta_hats_bstrap, dims = 2)) .* quantile(TDist(b_reps - 1), 1 - alpha/2)  #tries to compensate for narrow CIs
      return tuple.(theta_hat .- sqrt(b_reps/(b_reps - 1)).*reshape(collect(quantiles), no_of_params),  theta_hat .+ sqrt(b_reps/(b_reps - 1)).*reshape(collect(quantiles), no_of_params))
    else
      throw(ArgumentError("Must provide viable type of bootstrap interval."))
    end
  end
end
