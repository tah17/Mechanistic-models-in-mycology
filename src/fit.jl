using Distributed
using PyCall

"""
    FitType

Struct for fitting algorthims and arguments.
"""
abstract type FitType end

"""
    PyOptimiser

Struct for the SciPy fitting algorthims and arguments.
"""
abstract type PyOptimiser <: FitType end

"""
    LocalPyOptimiser

Local SciPy optimiser which just holds the python function, `pyfun` and the initial parameter guesses, `p0`, for optimisation.
Can also specify the number of iterations the optimiser will run for, `niter`.
"""
struct LocalPyOptimiser <: PyOptimiser
    pyfun::PyObject
    p0::AbstractArray{Float64,1}
    niter::Union{Int64, Nothing}
end

#
# If niters are not needed in the algorithm then they are not specified and set to nothing
#
LocalPyOptimiser(pyfun::PyObject, p0::AbstractArray{Float64,1}) = LocalPyOptimiser(pyfun, p0, nothing)

"""
    GlobalPyOptimiser

Global SciPy optimiser which just holds the python function, `pyfun` adn the parameter range, `p_range`, for optimisation.
"""
struct GlobalPyOptimiser <: PyOptimiser
    pyfun::PyObject
    p_range::AbstractArray{Tuple{T,T},1} where T <: Real
end

#
# PyOptimiser constructors
#
PyOptimiser(pyfun::PyObject, p0::AbstractArray{Float64,1}, niter::Union{Int64, Nothing}) = LocalPyOptimiser(pyfun, p0, niter)
PyOptimiser(pyfun::PyObject, p0::AbstractArray{Float64,1}) = LocalPyOptimiser(pyfun, p0)
PyOptimiser(pyfun::PyObject, p_range::AbstractArray{Tuple{T,T},1} where T <: Real) = GlobalPyOptimiser(pyfun, p_range)
