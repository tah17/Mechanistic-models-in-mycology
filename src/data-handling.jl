using DataFrames
include("../src/ODEs.jl")
include("../src/functions.jl")

"""
    get_fake_data(real_data::DataFrame, exp_cond::String, ode_species::Array{String,1}, ODE::Function, p::AbstractArray{Float64,1}, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64; <keyword arguments>)

Creates a 3D array of data using an ODE and parameter set, for a specific experimetnal condition e.g. low fungal inoculum.
The data has the same number of experiments as the real dataframe and uses the same time points as in the real dataframe.

#Arguments
- `real_data`: Data you want to emulate,
- `exp_cond`: The experimental condition you want to generate fake data for,
- `ode_species`: 1D array of ODE species names,
- `ODE`: ODE function,
- `p`: 1D array of kinetic parameters,
- `solver`: Solver used to solve the ODE,
- `saveat`: The time step size for solving;
- `x0`:  If known, the initial conditions of the ODE,
- `maxiters`: maximum iterations to solve your ODE for,
- `cb`: Call back function (if used) for the ODE.
"""
function get_fake_data(real_data::DataFrame,
                       exp_cond::String,
                       ode_species::Array{String,1},
                       ODE::Function,
                       p::AbstractArray{Float64,1},
                       solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                       saveat::Float64;
                       x0::Union{AbstractArray{Float64,1}, Missing} = missing,
                       maxiters::Union{Int64, Nothing} = nothing,
                       cb::Union{Nothing, Any} = nothing)

    data_cond = filter(x -> x.Condition == exp_cond, real_data)
    if size(data_cond)[1] == 0
        throw(ArgumentError("Experimental condition $exp_cond is not in the data set provided."))
    end
    t_pts = sort(unique(data_cond.Time))
    max_no_exps = length(unique(data_cond.Experiment))
    no_of_species = length(ode_species)
    sim_dat = Array{Union{Missing,Float64},3}(missing, length(t_pts), convert(Int64, max_no_exps), no_of_species)
    sims = simulate(ODE, p, (0.0, t_pts[end]), solver, saveat; no_of_species = no_of_species, x0 = x0, noise = false, maxiters = maxiters, cb = cb)
    if ismissing(sims)  #filters out parameters ----> unsolveable ODEs
        return sim_dat
    else
        for i in 1:no_of_species
            tmp = vcat([sims[i, findfirst(x -> isapprox(x, t, rtol = 0, atol = saveat - saveat/10), sims.t)] for t in t_pts]...)   #picks out the t_pts wanted from the ODE solution
            sim_dat[:, :, i] = repeat(tmp, 1, convert(Int64, max_no_exps))
        end
        return sim_dat
    end
end

"""
    get_fake_data(real_data::DataFrame, exp_cond::String, ode_species::Array{String,1}, sims::ODESolution, p::AbstractArray{T,1} where T<:Any, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64; <keyword arguments>)

Creates a 3D array of data using an ODE and parameter set, for a specific experimetnal condition e.g. low fungal inoculum.
The data has the same number of experiments as the real dataframe and uses the same time points as in the real dataframe.

#Arguments
- `real_data`: Data you want to emulate,
- `exp_cond`: The experimental condition you want to generate fake data for,
- `ode_species`: 1D array of ODE species names,
- `sims`: ODE solution object,
- `p`: 1D array of kinetic parameters,
- `solver`: Solver used to solve the ODE,
- `saveat`: The time step size for solving;
- `x0`:  If known, the initial conditions of the ODE,
- `maxiters`: maximum iterations to solve your ODE for,
- `cb`: Call back function (if used) for the ODE.

"""
function get_fake_data(real_data::DataFrame,
                       exp_cond::String,
                       ode_species::Array{String,1},
                       sims::ODESolution,
                       p::AbstractArray{T,1} where T<:Any,
                       solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                       saveat::Float64;
                       x0::Union{AbstractArray{Float64,1}, Missing} = missing,
                       maxiters::Union{Int64, Nothing} = nothing,
                       cb::Union{Nothing, Any} = nothing)

    data_cond = filter(x -> x.Condition == exp_cond, real_data)
    if size(data_cond)[1] == 0
        throw(ArgumentError("Experimental condition $exp_cond is not in the data set provided."))
    end
    t_pts = sort(unique(data_cond.Time))
    max_no_exps = length(unique(data_cond.Experiment))
    no_of_species = length(ode_species)
    sim_dat = Array{Any,3}(missing, length(t_pts), convert(Int64, max_no_exps), no_of_species)
    if ismissing(sims)  #filters out parameters ----> unsolveable ODEs
        return sim_dat
    else
        for i in 1:no_of_species
            tmp = vcat([sims[i, findfirst(x -> isapprox(x, t, rtol = 0, atol = saveat - saveat/10), sims.t)] for t in t_pts]...)
            sim_dat[:, :, i] = repeat(tmp, 1, convert(Int64, max_no_exps))
        end
        return sim_dat
    end
end

"""
    get_fake_data(ODE::Function, ode_species::Array{String,1}, p::Union{AbstractArray{Float64,1}, AbstractArray{Any,1}}, saveat::Real, t_pts::AbstractArray{T,1} where T <: Real, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, x0::Union{AbstractArray{Float64,1}, Missing}, no_exps::Int64; <keyword arguments>)

Creates a 3D array of data using an ODE and parameter set, for a specific experimetnal condition, number of experiments and time points.

#Arguments
- `ODE`: ODE function,
- `ode_species`: 1D array of ODE species names,
- `p`: 1D array of kinetic parameters,
- `solver`: Solver used to solve the ODE,
- `t_pts`: Time points you want to generate the fake data for,
- `saveat`: The time step size for solving;
- `x0`:  If known, the initial conditions of the ODE,
- `maxiters`: maximum iterations to solve your ODE for,
- `cb`: Call back function (if used) for the ODE.
"""
function get_fake_data(ODE::Function,
                       ode_species::Array{String,1},
                       p::Union{AbstractArray{Float64,1}, AbstractArray{Any,1}},
                       saveat::Real,
                       t_pts::AbstractArray{T,1} where T <: Real,
                       solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                       x0::Union{AbstractArray{Float64,1}, Missing},
                       no_exps::Int64;
                       maxiters::Union{Int64, Nothing} = nothing,
                       cb::Union{Nothing, Any} = nothing)

  no_of_species = length(ode_species)
  sim_dat = Array{Union{Missing,Float64},3}(missing, length(t_pts), convert(Int64, no_exps), length(ode_species))
  sims = simulate(ODE, p, (0.0, t_pts[end]), solver, saveat; no_of_species = no_of_species, x0 = x0, noise = false, maxiters = maxiters, cb = cb)
  if ismissing(sims)
      return sim_dat
  else
      for i in 1:no_of_species
          tmp = vcat([sims[i, findfirst(x -> isapprox(x, t, rtol = 0, atol = saveat - saveat/10), sims.t)] for t in t_pts]...)
          sim_dat[:, :, i] = repeat(tmp, 1, convert(Int64, no_exps))
      end
      return sim_dat
    end
end

"""
    add_noise(fake_species_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T}, noise_level::String, real_data::DataFrame, exp_cond::String)

Given a dictionary (`fake_species_data`) matching the a species string (e.g "F") to a 2D data set for a given experimental condtion (`exp_cond`)
`add_noise` adds a `noise_level` (e.g. "low") to the 2D data sets, which is calculated based on the variance of the `real_data`.

"""
function add_noise(fake_species_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T},
                   noise_level::String,
                   real_data::DataFrame,
                   exp_cond::String;
                   RNG::Union{MersenneTwister, Nothing} = nothing)

    data = filter(x -> !ismissing(x.Reading), real_data)
    tmp_df = filter(x -> x.Condition == exp_cond, data)
    if size(tmp_df)[1] == 0
        throw(ArgumentError("Experimental condition $exp_cond is not in the data set provided."))
    end
    df = by(tmp_df, :Species) do df
        DataFrame(m = mean(df[:Reading]), s² = var(df[:Reading]), max = maximum(df[:Reading]))
    end
    current_vars = df.s²
    if noise_level == "none"
        chosen_var = 0.0.*current_vars
    elseif noise_level == "low"
        chosen_var =  current_vars ./ 4
    elseif noise_level == "med"
        chosen_var = current_vars
    elseif noise_level == "high"
        chosen_var =  current_vars .* 4
    elseif noise_level == "max"
        chosen_var = df.max .^ 2
    else
        throw(ArgumentError("$noise_level is an invalid noise input."))
    end
    if isnothing(RNG)
        noisy_data = map(x -> x[2] .+ sqrt(chosen_var[x[1]]).*rand(Normal(), size(x[2])), enumerate(values(fake_species_data)))   #cannot use map! as need the index
    else
        noisy_data = map(x -> x[2] .+ sqrt(chosen_var[x[1]]).*rand(RNG, Normal(), size(x[2])), enumerate(values(fake_species_data)))   #cannot use map! as need the index
    end
    for elt in noisy_data
        elt[findall(x -> x < 0, elt)] .= 0  #fungal burden, cytokines, and neutrophils cannot be negative
    end
    return Dict(zip(keys(fake_species_data), noisy_data))
end

"""
    add_noise(fake_species_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T}, noise_level::String, current_vars::Array{T,1} where T <: Real, data_maxs::Array{T,1} where T <: Real)

Given a dictionary (`fake_species_data`) matching the a species string (e.g "F") to a 2D data set for a given experimental condtion (`exp_cond`)
`add_noise` adds a `noise_level` (e.g. "low") to the 2D data sets, which is calclated using the variance (`current_vars`) and maximum values (`data_maxs`)
that the user specifies.

"""
function add_noise(fake_species_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T},
                   noise_level::String,
                   current_vars::Array{T,1} where T <: Real;
                   data_maxs::Union{Array{T,1} where T <: Real, Nothing} = nothing,
                   RNG::Union{MersenneTwister, Nothing} = nothing)

    if length(fake_species_data) != length(current_vars)
        throw(ArgumentError("Need to provide the same number of variances ($(length(current_vars))) as data ($(length(fake_species_data)))."))
    end
    if isnothing(data_maxs) && noise_level == "max"
        throw(ArgumentError("For noise_level = $noise_level maximums levels for each species need to be specified."))
    end
    if any(current_vars .< 0.0)
        missing_arr = map(x -> Array{Missing,2}(missing, size(x)), values(fake_species_data))
        return Dict(zip(keys(fake_species_data), missing_arr))
    end
    if noise_level == "none"
        chosen_var = 0.0.*current_vars
    elseif noise_level == "low"
        chosen_var =  current_vars ./ 4
    elseif noise_level == "med"
        chosen_var = current_vars
    elseif noise_level == "high"
        chosen_var =  current_vars .* 4
    elseif noise_level == "max"
        chosen_var = data_maxs .^ 2
    else
        throw(ArgumentError("$noise_level is an invalid noise input."))
    end
    if isnothing(RNG)
        noisy_data = map(x -> x[2] .+ sqrt(chosen_var[x[1]]).*rand(Normal(), size(x[2])), enumerate(values(fake_species_data)))   #cannot use map! as need the index
    else
        noisy_data = map(x -> x[2] .+ sqrt(chosen_var[x[1]]).*rand(RNG, Normal(), size(x[2])), enumerate(values(fake_species_data)))
    end
    for elt in noisy_data
        elt[findall(x -> x < 0, elt)] .= 0  #fungal burden, cytokines, and neutrophils cannot be negative
    end
    return Dict(zip(keys(fake_species_data), noisy_data))
end

"""
    emulate_sparsity(noisy_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T}, real_data::DataFrame, exp_cond::String)

Emulates the data sparsity of the real data (`real_data`) for a given experimental condition, `exp_cond`, in the fake data generated, where the fake data provided is a dictionary of simulated data
with keys of the ODE species (`noisy_data`). Returns a data frame.

"""
function emulate_sparsity(noisy_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T},
                          real_data::DataFrame,
                          exp_cond::String)

    data_cond = filter(x -> x.Condition == exp_cond, real_data)
    if size(data_cond)[1] == 0
        throw(ArgumentError("Experimental condition $exp_cond is not in the data set provided."))
    end
    t_pts = sort(unique(data_cond.Time))
    max_no_exps = length(unique(data_cond.Experiment))
    for (idx, elt) in enumerate(groupby(data_cond, :Species))
        undef_readings = Array{Missing,2}(missing, length(t_pts), max_no_exps)
        tmp1 = get(noisy_data, keys(groupby(data_cond, :Species))[idx][1], undef_readings)  #if the species is not in the real data - return an array of missing values
        if all(ismissing.(tmp1))
            throw(ArgumentError("The species $(keys(groupby(data_cond, :Species))[idx][1]) in the Dataframe is not in the Dict provided"))
        end
        tmp2 = vcat([[tmp1[findfirst(x -> x == t, t_pts), idx] for t in x.Time] for (idx, x) in enumerate(groupby(elt, :Experiment))]...)  #emulates the sparsity of the dataframe
        elt.Reading = tmp2
    end
    return data_cond
end

"""
    make_df(species_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T}, t_pts::Array{Float64,1}, exp_cond::String; starting_exp::Int64 = 0)

Create a dataframe given a dictionary matching the species to a 2D array (`species_data`), the time points of the data frame `t_pts` and
the experiment number the data frame begins at (`starting_exp`).

Intended to be used following function calls from `get_fake_data` or `add_noise`.
"""
function make_df(species_data::Union{Dict{String,Array{T,2} where T}, Dict{String,Array{T,2}} where T},
                 t_pts::Array{Float64,1},
                 exp_cond::String;
                 starting_exp::Int64 = 0)

    dfs = [DataFrame() for elt in species_data]
    no_exps = Int64.(zeros(length(species_data) + 1))
    for (idx, elt) in enumerate(species_data)
        no_exps[idx + 1] = convert(Int64, length(elt[2])/length(t_pts))  #write argument error to ensure this won't fail
        dfs[idx] = DataFrame(Time = repeat(t_pts, outer = no_exps[idx + 1]),
                             Reading = vcat(elt[2]...),
                             Experiment = repeat(["E$i" for i in starting_exp + sum(no_exps[1:idx]) + 1:starting_exp + sum(no_exps[1:idx+1])], inner = length(t_pts)),
                             Species = repeat([elt[1]], inner = length(elt[2])),
                             Condition = repeat([exp_cond], inner = length(elt[2])))
    end
    if length(species_data) > 1
        return join([df for df in dfs]..., kind  = :outer, on = intersect([names(df) for df in dfs]...))
    else
        return dfs[1]
    end
end


"""
    residuals(ODE::Function, user_defined::Function, dat::DataFrame, ode_species::Array{String,1}, p::Union{AbstractArray{Float64,1}, AbstractArray{Any,1}}, Tspan::Tuple{Float64,Float64}, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64; x0::Union{AbstractArray{Float64,1}, Missing} = missing, noise::Bool = false, maxiters::Union{Int64, Nothing} = nothing, cb::Union{Nothing, Any} = nothing)

Given experimental data in a data frame, calculates the sum of squared errors between the data and
the ODE output for a chosen parameter, `p`. If the ODE cannot be solved for the value of `p` provided
then the sum of squared errors is returned between the data and an intercept at 0 (worst case scenario).

#Arguments
- `ODE`: ODE function.
- `user_defined`:,
- `dat`: The real data,
- `ode_species`: 1D array of ODE species names,
- `p`: 1D array of kinetic parameters,
- `Tspan`: Time span the ODE should be solved for,
- `solver`: Solver used to solve the ODE,
- `saveat`: The time step size for solving,
- `x0`: If known, the initial conditions of the ODE,
- `noise`: If the noise is estimated or not,
- `maxiters`: Maximum iterations to solve your ODE for,
- `cb`: Call back function (if used) for the ODE.

#Returns
- 0.5 x Sum of squared errors between fake data generated by a parameter value `p` and the real data `dat`.
"""
function residuals(ODE::Function,
                   user_defined::Function,
                   dat::DataFrame,
                   ode_species::Array{String,1},
                   p::Union{AbstractArray{Float64,1}, AbstractArray{Any,1}},
                   Tspan::Tuple{Float64,Float64},
                   solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                   saveat::Float64;
                   x0::Union{AbstractArray{Float64,1}, Missing} = missing,
                   noise::Bool = false,
                   maxiters::Union{Int64, Nothing} = nothing,
                   cb::Union{Nothing, Any} = nothing)

  res = 0
  t_pts = sort(unique(dat.Time))
  no_of_species = length(ode_species)
  for (idx, elt) in enumerate(unique(dat.Condition))
      sim_species_data = user_defined(dat, elt, ode_species, ODE, p, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
      measured_species = length(keys(sim_species_data))
      var_used = p[length(p) - measured_species+1:end]
      data_cond = filter(x -> x.Condition == elt, dat)
      if all(ismissing.(vcat([get(sim_species_data, k, missing) for k in keys(sim_species_data)]...)))  #checks if the ODE could be solved
          res += sum((data_cond.Reading).^2)
      else
          if noise && any(var_used .== 0.0)
              throw(ArgumentError("Cannot have a zero noise variance. You specified $var_used variances"))
          elseif noise
              noisy_data = add_noise(sim_species_data, "med", var_used)  #defaults at "med" level as this simulates data with the supplied variance
              if all(ismissing.(vcat([get(noisy_data, k, missing) for k in keys(noisy_data)]...)))  #checks if the negative noise was passed
                  return res += sum((data_cond.Reading).^2)
              end
              fake_species_data = emulate_sparsity(noisy_data, dat, elt)
              for (idx, elt) in enumerate(groupby(data_cond, :Species))
                  sim_dat = filter(x -> x.Species == unique(elt.Species)[1], fake_species_data)
                  res += sum((elt.Reading - sim_dat.Reading).^2)./(var_used[idx])
              end
          else
              fake_species_data = emulate_sparsity(sim_species_data, dat, elt)
              res += sum((data_cond.Reading - fake_species_data.Reading).^2)
          end
      end
  end
  return 0.5*res
end
