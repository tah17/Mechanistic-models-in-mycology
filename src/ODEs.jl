using DifferentialEquations

"""
    TotalDEInput

Structure which holds the inputs needed for computing your ODE.

# Fields
- `DE::Function`: The ODE function,
- `tspan::Tuple{Float64,Float64}`: The timespan the DE should be solved for,
- `solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm`: The `OrdinaryDiffEq` solver you want to use to solve your ODE,
- `saveat::Float64`: The time step size for solving,
- `x0::Union{AbstractArray{Float64,1}, Missing}`: If known, the initial conditions of the ODE,
- `no_of_species::Int64`: Number of species in ODE.
"""
struct TotalDEInput
    DE::Function
    tspan::Tuple{Float64,Float64}
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm
    saveat::Float64
    x0::Union{AbstractArray{Float64,1}, Missing}
    no_of_species::Int64
    function TotalDEInput(o, t, sol, sav, x, n)
        if t[2] < t[1]
            throw(ArgumentError("Your end time point, $(t[2]), must be larger than your initial time point, $(t[1])."))  #checks time points are well posed
        end
        if !ismissing(x)
            if length(x) != n
                throw(ArgumentError("You gave $(length(x)) initial conditions but stated that there are $n species"))   #checks i.c.s cover all vars in the ODE
            end
        end
        new(o, t, sol, sav, x, n)
    end
end

TotalDEInput(DE::Function, tspan::Tuple{Float64,Float64}, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, saveat::Float64, no_of_species::Int64) = TotalDEInput(DE, tspan, solver, saveat, missing, no_of_species)

"""
    summarise(ODE::TotalDEInput)

Summarises the ODE.
"""
function summarise(ODE::TotalDEInput)
    println("ODE: $(ODE.DE)")
    println("Time span: $(ODE.tspan)")
    println("Solver: $(ODE.solver)")
    println("Saved step size: $(ODE.saveat)")
    println("Initial Conditions: $(ODE.x0)")
    println("Number of species: $(ODE.no_of_species)")
end

# ODEs commomly used to avoid having to define them in each script.
# DifferentialEquations v6.8.0

function Example_ODE(dx, x, params, t)
#
#
#
#
end

#
#for unit testing
#
function sigmoid(dx, x, par, t)
  dx[1] = par[1]*x[1]*(1 - x[1])
end

function linear_function(dx, x, par, t)
  dx[1] = par[1]
end
