using DifferentialEquations
using Distributions
using ForwardDiff
using LinearAlgebra
using Distributed
using DataFrames

include("ODEs.jl")
include("misc.jl")

"""
    y_hat_prob(ODE::Function, p::AbstractArray{Float64,1}, Tspan::Tuple{Float64,Float64}; no_of_species::Union{Int64, Nothing} = nothing, x0::Union{AbstractArray{Float64,1}, Missing} = missing, noise::Bool = false)

Takes a parameter set and the same arguments needed for specifying an ODE as in the package
`DifferentialEquations.jl` and returns an `ODEProblem` to be used with the `ForwardDiff`
package. The idea is to then call this function in the following form:

```
function y_hat
  _prob = remake(prob; u0 = convert.(eltype(params), prob.u0), p = params)
  sol = solve(_prob, solver, saveat = saveat)
  foo(sol)
end
```

where `foo()` is a function that returns a 1D array of vertically concatenated time series.
If noise is specified to be true, then the final parameter is taken to be ``\\sigma`` in
``y \\sim \\mathcal{N}(f, \\sigma^2)``, where ``y`` is the data and ``f`` is the ODE.
If `x_0` is not declared the function assumes the first `n` parameters are the initial conditions.

#Arguments
- `ODE`: Function,
- `p`: Array of parameters,
- `Tspan`: Timespan,
- `no_of_species`: number of species in the ODE,
- `x0`: Initial conditions of ODE,
- `noise`: If noise is simulated

#Return
Obs: `ODEProblem`.
"""
function y_hat_prob(ODE::Function,
                    p::AbstractArray{Float64,1},
                    Tspan::Tuple{Float64,Float64};
                    no_of_species::Union{Int64, Nothing} = nothing,
                    x0::Union{AbstractArray{Float64,1}, Missing} = missing,
                    noise::Bool = false)

  if no_of_species == nothing  #optional arguments added for kwargs
    throw(ArgumentError("Must provide the number of species: 'no_of_species'."))
  end
  #seperate kinetic parameters from external noise, if noise is specified to be true ∴ parameter order matters
  if noise
    p, σ = p[1:length(p) - 1], p[end]
    if σ < 0
      throw(ArgumentError("The noise must be positive, you specifed a noise of $σ < 0"))
    end
  else
    p = p
  end
  #seperate kinetic parameters from initial coniditions, if x0 is missing
  if x0 === missing
    x0 = Array{Float64,1}(undef, no_of_species)
    x0, p = p[1:no_of_species], p[no_of_species+1:end]
  else
    p = p
  end
  return ODEProblem(ODE, x0, Tspan, p)
end

"""
    FIM_i(θ::AbstractArray{Float64,1}, i::Union{Int64, Float64}, y_hat::Function, y::AbstractArray{Float64,1}, sigma::Bool)

Calculates the Observed Fisher Information Matrix (FIM) of a function (`y_hat`) at a specific
time point (``i``) with respect to the parameters ``\\theta = [p, \\sigma]``, if `sigma` and ``\\theta = [p]`` if `sigma == false`.
The intermediate calculations are the following:
    `(transpose(J_i)*J_i .+ (y_hat(θ[1:end-1])[i] - y[i])*H_i)./θ[end]^2`: FIM w.r.t. to the parameters, `p` (size `n-1 x n-1` when ``\\theta`` is of size `n`),
    `(y_hat(θ[1:end-1])[i] - y[i]).^2`: the diagnonal entry of the FIM w.r.t. to the noise, ``\\sigma`` (size `1 x 1`), and,
    `(y_hat(θ[1:end-1])[i] - y[i])*J_i`: the entries of the FIM which correspond to ``\\frac{dl}{d\\sigma dp}`` (`dl/dpdσ`) (and hence `dl/dσdp`), where `l` is the log likelihood (size `n-1 x 1`).
The final FIM can then be constructed as follows:
    `F[end, 1:end-1] = (2/(θ[end]^3)).*(y_hat(θ[1:end-1])[i] - y[i])*J_i`
    `F[1:end-1, end] = (2/(θ[end]^3)).*(y_hat(θ[1:end-1])[i] - y[i])*J_i`
    `F[end, end] = (-3/(θ[end]^4))*(y_hat(θ[1:end-1])[i] - y[i]).^2`

#Arguments
- `θ`: 1D array of your parameters ``\\theta = [p, \\sigma]``,
- `i`: Specific time point (``i``),
- `y_hat`: Function,
- `y`: data,
- `sigma`: If you are estimating the additive experimental noise, ``\\sigma`` in the Normal distribution, or not.

#Return
Observed FIM at time point/data point `i` with resulting size `(length(θ), length(θ))`.
"""
function FIM_i(θ::AbstractArray{Float64,1},
               i::Union{Int64, Float64},
               y_hat::Function,
               y::Union{AbstractArray{Float64,1}, AbstractArray{Union{Missing,Float64},1}},
               sigma::Bool)

  if ismissing(y[i]) || ismissing(y_hat(θ))  #if the data or dual ODE solution is missing then nothing array to avoid having to run into handling of missing value errors
    return zeros(length(θ), length(θ))  #this will not change FIM calculation since zero is identity of addition
  elseif sigma #if you estimating FIM for the noise term σ
    J_i = ForwardDiff.gradient(x -> y_hat(x)[i], θ[1:end-1])  #Jacobian w.r.t. to p
    H_i = ForwardDiff.hessian(x -> y_hat(x)[i], θ[1:end-1])  #Hessian w.r.t. to p
    F = Array{Float64,2}(undef, length(θ), length(θ))  #initialise FIM matrix
    F[1:end-1, 1:end-1] = (transpose(J_i)*J_i .+ (y_hat(θ[1:end-1])[i] - y[i])*H_i)./θ[end]^2  #FIM w.r.t. to the parameters, p
    F[end, 1:end-1] = (2/(θ[end]^3)).*(y[i] - y_hat(θ[1:end-1])[i])*J_i  #the entries of the FIM which correspond to dl/dpdσ (and hence dl/dσdp), where l is the log likelihood
    F[1:end-1, end] = (2/(θ[end]^3)).*(y[i] - y_hat(θ[1:end-1])[i])*J_i  #the entries of the FIM which correspond to dl/dpdσ (and hence dl/dσdp), where l is the log likelihood
    F[end, end] = (-1/θ[end]^2) + (3/(θ[end]^4))*(y[i] - y_hat(θ[1:end-1])[i]).^2  #the diagnonal entry of the FIM w.r.t. to the noise, σ
  else  #add in try catch for missing encountered
    J_i = ForwardDiff.gradient(x -> y_hat(x)[i], θ)  #Jacobian w.r.t. to p
    H_i = ForwardDiff.hessian(x -> y_hat(x)[i], θ)  #Hessian w.r.t. to p
    F = (transpose(J_i)*J_i .+ (y_hat(θ)[i] - y[i])*H_i)  #FIM w.r.t. to the parameters, p
  end
  return F
end

"""
    FIM(θ::AbstractArray{Float64,1}, y::AbstractArray{Float64,1}, t_pts::Union{AbstractArray{Float64,1}, AbstractArray{Int64,1}}, ODE::Function, Tspan::Tuple{Float64,Float64}, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64; no_of_species::Union{Int64, Nothing} = nothing, x0::Union{AbstractArray{Float64,1}, Missing} = missing, noise::Bool = false)

Takes a particular parameter set, ``\\theta_n``, and calculates ``\\mathcal{I}_n(\\theta_n)`` the observed Fisher Information Matrix (FIM).

#Arguments:
- `θ`: 1D array of your parameters ``\\theta = [p, \\sigma]``,
- `y`: Observed data,
- `t_pts`: Timepoints of the observed data,
- `ODE`: ODE function,
- `Tspan`: Time span to solve the ODE for,
- `solver`: Solver used to solve the ODE,
- `saveat`: The time step size for solving,
- `no_of_species`: The number of species in the ODE,
- `x0`: If known, the initial conditions of the ODE,
- `noise`: If the noise is estimated or not,
- `CI`: Specifies type of return: confidence interval (C.I.) or not,
- `alpha`: Confidence level for the C.I.

#Returns
The observed FIM is returned, which is of size `(length(θ), length(θ))`, or, a FIM based confidence interval centered (Tuple) around the parameter estimate provided (`θ`).
"""
function FIM(θ::AbstractArray{Float64,1},
             user_defined::Function,
             dat::DataFrame,
             ODE::Function,
             Tspan::Tuple{Float64,Float64},
             solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
             saveat::Float64;
             no_of_species::Union{Int64, Nothing} = nothing,
             x0::Union{AbstractArray{Float64,1}, Missing} = missing,
             noise::Bool = false,
             maxiters::Union{Int64, Nothing} = nothing,
             cb::Union{Nothing, Any} = nothing)

  if noise && θ[end] < 0  #will not allow a negative noise to be passed.
      return missing
  end
  t_pts = sort(unique(dat.Time))
  y = dat.Reading
  no_of_exps = length(unique(dat.Experiment))
  prob = y_hat_prob(ODE, θ, Tspan, no_of_species = no_of_species, x0 = x0, noise = noise)  #sets up the ODE
  function y_hat(params)  #function which solves the ODE in dual numbers, a format needed for the FIM_i function.
                          #... & then returns the stacked ODE solutions at the time points specified in t_pts.
      _prob = remake(prob; u0 = convert.(eltype(params), prob.u0), p = params)
      sol = solve(_prob, solver, saveat = saveat)
      if !isnothing(maxiters) && !isnothing(cb)
          sol = solve(_prob, solver, saveat = saveat, maxiters = maxiters, callback = cb)
      elseif !isnothing(maxiters)
          sol = solve(_prob, solver, saveat = saveat, maxiters = maxiters)
      elseif !isnothing(cb)
          sol = solve(_prob, solver, saveat = saveat, callback = cb)
      else
          sol = solve(_prob, solver, saveat = saveat)
      end
      if sol.retcode == :Success  #if the ODE solves (tmp below) does not mean it's dual will - this checks dual is solveable
          fake_y = []
          for (idx, elt) in enumerate(unique(dat.Condition))
              sim_species_data = user_defined(dat, elt, ode_species, sol, params, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
              if all(ismissing.(vcat([get(sim_species_data, k, missing) for k in keys(sim_species_data)]...)))   #if ODE unsolveable ----> missing
                  return missing
              else
                  data_cond = filter(x -> x.Condition == elt, dat)
                  max_no_exps = length(unique(data_cond.Experiment))
                  t_pts = sort(unique(data_cond.Time))
                  for (idy, elty) in enumerate(groupby(data_cond, :Species))
                      undef_readings = Array{Missing,2}(missing, length(t_pts), max_no_exps)
                      tmp1 = get(sim_species_data, keys(groupby(data_cond, :Species))[idy][1], undef_readings)
                      tmp2 = vcat([[tmp1[findfirst(x -> x == t, t_pts), idz] for t in z.Time] for (idz, z) in enumerate(groupby(elty, :Experiment))]...)
                      push!(fake_y, tmp2)
                  end
              end
          end
          return collect(vcat(fake_y...))
      else
          return missing
      end
  end
  tmp = solve(prob, solver, saveat = saveat)
  if tmp.retcode == :Success  #ensures that the ODE is solved for all the time points in t_pts.
      F_obs = @sync @distributed (+) for i in 1:length(y)
          FIM_i(θ, i, y_hat, y, noise);
      end
      return F_obs
  else
      @warn "The parameter values $θ lead to an unsolveable ODE"
      return Array{Missing,2}(missing, (length(θ), length(θ)))
  end
end

function FIM(θ::AbstractArray{Float64,1},
             user_defined::Function,
             dat::DataFrame,
             ODE::TotalDEInput;
             noise::Bool = false,
             maxiters::Union{Int64, Nothing} = nothing,
             cb::Union{Nothing, Any} = nothing)

    return FIM(θ, user_defined, dat, ODE.DE, ODE.tspan, ODE.solver, ODE.saveat; no_of_species = ODE.no_of_species, x0 = ODE.x0, noise = noise)
end
