using DifferentialEquations
using Distributions
using XLSX
using ForwardDiff
using LinearAlgebra
using Distributed
using ScikitLearn

include("ODEs.jl")

"""
    simulate(ODE::Function, p::AbstractArray{Float64,1}, Tspan::Tuple{Float64,Float64}, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64; no_of_species::Union{Int64, Nothing} = nothing, x0::Union{AbstractArray{Float64,1}, Missing} = missing, noise::Bool = false)

Takes a parameter set and the same arguments needed for specifying an ODE as in the package
`DifferentialEquations.jl` and simulates a given ODE. If noise is specified to be true, then
the final parameter is taken to be ``\\sigma`` in ``y \\sim \\mathcal{N}(f, \\sigma^2)``, where ``y`` is the data and ``f`` is the ODE.
If `x_0` is not declared the function assumes the first n parameters are the initial conditions.
If the ODE cannot be solved to the specified time scale "`missing`" will be returned.

#Arguments
- `ODE`: ODE function,
- `p`: 1D array of kinetic parameters,
- `Tspan`: Time span to solve the ODE for,
- `solver`: Solver used to solve the ODE,
- `saveat`: The time step size for solving,
- `no_of_species`: Number of species in ODE,
- `x0`:  If known, the initial conditions of the ODE,
- `noise`: If the noise is estimated or not.

#Returns
- `Obs::DESolution` or
- `missing`
"""
function simulate(ODE::Function,
                  p::AbstractArray{T,1} where T<:Real ,
                  Tspan::Tuple{Float64,Float64},
                  solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                  saveat::Float64;
                  no_of_species::Union{Int64, Nothing} = nothing,
                  x0::Union{AbstractArray{Float64,1}, Missing} = missing,
                  noise::Bool = false,
                  maxiters::Union{Int64, Nothing} = nothing,
                  cb::Union{Nothing, Any} = nothing)

  if no_of_species == nothing  #optional arguments added and then ArgumentError thrown is missing for kwargs
      throw(ArgumentError("Must provide the number of species: 'no_of_species'."))
  end
  #seperate kinetic parameters from external noise, if noise is specified to be true
  if noise && no_of_species >= length(p)
      throw(ArgumentError("Parameters `p` misspecified.
                           Noise is specified by including the variance of noise for each species at the end of the parameter vector `p`
                           that the kinetic parameters are inputted in."))
  elseif noise
      p, p_σ = convert(Array{Float64,1}, p[1:length(p) - no_of_species]), p[(length(p) - no_of_species + 1):end]
  else
      p = p
  end
  #seperate kinetic parameters from initial coniditions, if x0 is missing
  if ismissing(x0) && no_of_species >= length(p)
      throw(ArgumentError("Parameters `p` misspecified.
                           Missing i.c. parameters are specified as parameters at the start of the parameter vector `p`
                           that the kinetic parameters are inputted in."))
  elseif ismissing(x0)
      x0, p = p[1:no_of_species], p[no_of_species+1:end]
  else
      p = p
  end
  prob = ODEProblem(ODE, x0, Tspan, p)
  if !isnothing(maxiters) && !isnothing(cb)
      Obs = solve(prob, solver, saveat = saveat, maxiters = maxiters, callback = cb)
  elseif !isnothing(maxiters)
      Obs = solve(prob, solver, saveat = saveat, maxiters = maxiters)
  elseif !isnothing(cb)
      Obs = solve(prob, solver, saveat = saveat, callback = cb)
  else
      Obs = solve(prob, solver, saveat = saveat)
  end
  if no_of_species != size(hcat(Obs.u...))[1]  #will cause issues when x0 is missing...
      throw(ArgumentError("You declared that no_of_species = $no_of_species but the ODE has $(size(hcat(Obs.u...))[1]) species."))
  end
  if noise  #add noise to solutions
      if all(p_σ .< zero(p_σ))
          @warn "The noise must be positive, you specifed a noise of $(p_σ[p_σ .< zero(p_σ)]) < 0"
          return missing
      end
      for i in 1:no_of_species
          Obs[i, :] = Obs[i, :] .+ sqrt(p_σ[i]).*rand(Normal(), length(Obs[i, :]))
      end
  end
  if Obs.retcode == :Success
      return Obs
  else
      return missing
  end
end

"""
    get_data_per_species(xf::AbstractArray{Any,2}, wanted_idxs::AbstractArray{UnitRange{Int64},1}, start_of_data::Int64;  na::String = "─")

Takes an already extracted excel sheet using `XLSX`, e.g:
`xf = XLSX.readxlsx("mypath/myspreadsheet.xlsx")["Sheet1"][:]`,
the indices which you want from this spreadsheet (in terms of columns) and what row the
data starts on (`start_of_data`). The columns you take should always begin with times - and
this is the only place where time appears (index 1). The function will then return an
`Array` of the preffered type (`Union` of `Float` and `missing`) to be used in further functions.
Intended to be used only on one datasheet (i.e. for only one species). The idea would be
to then place each of the 2D arrays for each species in a larger 3D array.

#Arguments
- `xf`: Extracted excel sheet using `XLSX` (2D array),
- `wanted_idxs`: Column indices wanted from the spreadsheet,
- `start_of_data`: Row index that the spreadsheet begins on,
- `na`: NA in the spreadsheet.

#Return
Array of the experimental data with size `(Number of time points x Number of Experiments)`.
"""
function get_data_per_species(xf::AbstractArray{Any,2},
                              wanted_idxs::AbstractArray{UnitRange{Int64},1},
                              start_of_data::Int64;
                              na::String = "─")

  xf_truncated = hcat([xf[start_of_data:end, idxs] for idxs in wanted_idxs]...)
  converted_xf = convert(Array{Union{Real, String},2}, xf_truncated)  #convert the array from type Any to call findall()
  df = Array{Union{Float64, Missing},2}(missing, size(converted_xf))
  for idx in findall(x -> x != na, converted_xf)  #where there is no NA value, update array with excel file value
      df[idx] = converted_xf[idx]
  end
  return df
end

"""
    get_data(xf::AbstractArray{Array{Any,2},1}, wanted_idxs::AbstractArray{Array{UnitRange{Int64},1},1}, start_of_data::AbstractArray{Int64,1}; nas::AbstractArray{String,1} = repeat(["─"], outer =  length(xf)))

This function takes arrays of your arguments that you would use for `get_data_per_species`:
    1. An array such as:` xf = [xf_1, xf_2, xf_3]`, where
    ```
    xf_1 = XLSX.readxlsx("mypath/myspreadsheet_species1.xlsx")["Sheet1"][:]
    xf_2 = XLSX.readxlsx("mypath/myspreadsheet_species2.xlsx")["Sheet1"][:]
    xf_3 = XLSX.readxlsx("mypath/myspreadsheet_species3.xlsx")["Sheet1"][:]
    ```
    (note this should be specified in the same order as in the ODE).

    2. An array of the the indices which you want from the corresponding spreadsheets (in terms of columns),
    where, again, the columns you specify should always begin with "times" - and this is the *only* place where time
    appears (index 1). So, e.g.
    `wanted_idxs = [wanted_idxs_of_species1, wanted_idxs_of_species2, wanted_idxs_of_species3]`.

    3. The row where the data starts on, from each spreadsheet :
    `start_of_data = [start_of_data_of_species1, start_of_data_of_species2, start_of_data_of_species3]`.

    4. Finally, the NAs are what constitutes as an NA in your spreadsheets:
    `nas = [na1, na2, na3]`, which will default as `["─", ... ,"─"]`.

This function then returns a 3D array of your experimental data in the form of `(Number of time points x Number of Experiments x Number of species)`,
which is the correct form to be used in the `residuals` function.

#Arguments
- `xf`: Array of the extracted excel sheets using `XLSX` (3D array),
- `wanted_idxs`: Array of column indices wanted from the spreadsheet,
- `start_of_data`: Array of the row indices that the spreadsheets begin on,
- `na`: Array of the NAs in the spreadsheets.

#Return
Array of the experimental data of size (`Number of time points x Number of Experiments x Number of species`).
"""
function get_data(xf::AbstractArray{Array{Any,2},1},  #find a better way to pass these inputs
                  wanted_idxs::AbstractArray{Array{UnitRange{Int64},1},1},
                  start_of_data::AbstractArray{Int64,1};
                  nas::AbstractArray{String,1} = repeat(["─"], outer =  length(xf)))

  max_no_exps = maximum([sum([length(x) for x in elt]) for elt in wanted_idxs]) - 1  #compute max no. of columns (excluding time) of the sheets passed
  unique_t = sort(unique(vcat([elt[start_of_data[idx]:end, 1] for (idx, elt) in enumerate(xf)]...)))  #takes all unique time points of all spreadsheets
  t_pts = convert(Array{Float64,1}, unique_t)
  dat = Array{Union{Float64, Missing},3}(missing, length(t_pts), max_no_exps + 1, length(xf))
  dat[:, 1, :] .= t_pts  #all of the 2D arrays in the 3D object begin with a time column
  for k in 1:size(dat)[3]
      dat_tmp = get_data_per_species(xf[k], wanted_idxs[k], start_of_data[k], na = nas[k])
      for (i, idxs) in enumerate(vcat([findall(x -> x == t, t_pts) for t in dat_tmp[:, 1]]...))
          dat[idxs, 2:size(dat_tmp)[2], k] .= dat_tmp[i, 2:end]
      end
  end
  return dat
end
