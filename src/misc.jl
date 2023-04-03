"""
    operation_on_missing(foo::Function, value::Any...)

Allows the user to conduct an operation on an variable that may be a Union{Any, Missing} type, where the function `foo` does not support `Missing` types.

#Arguments
- `value`: The variable you want to use foo() on but may contain `missing` values,
- `foo`: The function.

#Return:
`foo(value)` if `foo(skipmissing(value))` is supported, if not `missing` is returned. If all of `value` is `missing` only `missing` is returned.
"""
function operation_on_missing(foo::Function, value::Any...; kwargs...)
    if all(ismissing.(value...))  #if all the values in the "value" passed are missing then return just a missing value
        return missing
    elseif any(ismissing.(value...))  #if there is just the existence of a missing value then try and see if the function can already handle some missing values
        try
            foo(value...; kwargs...)
        catch err
            if err isa MethodError || err isa TypeError  #if not then see if you can get around this by enforcing that the function skips missing values
                try
                    foo(skipmissing(value...); kwargs...)
                catch err2
                    if err2 isa MethodError || err isa TypeError  #if using skipmissing() is not working then just return missing
                        return missing
                    else
                        rethrow(err)
                    end
                end
            else
                rethrow(err)
            end
        end
    else
        return foo(value...; kwargs...)  #if there are no missing values apply the function as normal
    end
end

"""
    get_values(ODE::DESolution, t_pts::Union{AbstractArray{Float64,1}, AbstractArray{Int64,1}}, saveat::Float64, species_idx::Int64)

Function that takes your solved ODE simulations for a species of interest and returns an `Array` of the ODE solutions at the user specified timepoints.

#Arguments
- `ODE`: ODE solutions (`DESolution`),
- `t_pts`: Time points of interest,
- `saveat`: The time step size used in ODE solving,
- `species_idx`: Index (in the ODE) of species.

#Return
Array of ODE simuations at wanted the time points.
"""
function get_values(ODE::Union{DESolution, Missing},
                    t_pts::Union{AbstractArray{Float64,1}, AbstractArray{Int64,1}},
                    saveat::Float64,
                    species_idx::Int64)

  if ismissing(ODE)
      return Array{Missing,1}(missing, length(t_pts))
  end
  if maximum(t_pts) > maximum(ODE.t) || minimum(t_pts) < minimum(ODE.t)
    throw(ArgumentError("The range that the ODE was solved for, ($(minimum(ODE.t)), $(maximum(ODE.t))), is smaller than the range of the time points provided, ($(minimum(t_pts)), $(maximum(t_pts)))."))
  end
  return vcat([ODE[species_idx, findfirst(x -> isapprox(x, t, rtol = 0, atol = saveat - saveat/10), ODE.t)] for t in t_pts]...)
end
