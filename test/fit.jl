using Test
using PyCall
using Random
using DifferentialEquations
using Distributions
Random.seed!(26922)
@pyimport scipy.optimize as so
include("../src/fit.jl")
include("../src/ODEs.jl")

@testset "Constructors Tests" begin
    #
    #arguments for ODE solving
    #
    Tspan = (0.0, 100.0)  #timspan
    x0 = [0.0]  #initial conditions
    solver = RK4()
    no_of_species = 1
    saveat = 1.0
    t_pts = collect(Tspan[1]:saveat:Tspan[end])
    niter = 100  #no of iterations needed for so.basinhopping
    pyfun = so.basinhopping  #scipy algorithms
    pyfun_alt = so.least_squares
    p0 = [rand(Normal(1, 0.1))]
    pyoptim = LocalPyOptimiser(pyfun, p0, niter)
    @test typeof(pyoptim) <: PyOptimiser
    newpyoptim = LocalPyOptimiser(pyfun, p0)
    @test newpyoptim.pyfun == pyoptim.pyfun
    @test newpyoptim.p0 == pyoptim.p0
    @test PyOptimiser(pyfun, p0, niter) == LocalPyOptimiser(pyfun, p0, niter)
    @test isa(PyOptimiser(pyfun, p0, niter), LocalPyOptimiser)
    globaloptim = PyOptimiser(pyfun, [(3, 5)])
    p_range = [(3, 5)]
    @test PyOptimiser(pyfun, p_range) == GlobalPyOptimiser(pyfun, p_range)
    ODE = TotalDEInput(linear_function, Tspan, solver, saveat, x0, no_of_species)
    @test ODE.no_of_species == no_of_species
    @test ODE.saveat == saveat
    @test ODE.solver == solver
    @test ODE.tspan == Tspan
    @test ODE.x0 == x0
    #
    # checks argument errors
    #
    @test_throws ArgumentError TotalDEInput(linear_function, (10.0, 5.0), solver, saveat, x0, no_of_species)
    @test_throws ArgumentError TotalDEInput(linear_function, Tspan, solver, saveat, x0, 2)
end
