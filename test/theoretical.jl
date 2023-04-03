using Distributed
addprocs(2);
@everywhere begin
    using Test
    using SharedArrays
    using PyCall
    using Random
    Random.seed!(1234)
end
include("../src/theoretical.jl")
include("../src/data-handling.jl")
include("../src/ODEs.jl")
@everywhere begin
    include(joinpath(@__DIR__, "../src/theoretical.jl"))
    include(joinpath(@__DIR__, "../src/data-handling.jl"))
    include(joinpath(@__DIR__, "../src/ODEs.jl"))
    @pyimport scipy.optimize as so
end
@testset "Fisher Information" begin
    #
    # testing y_hat_prob()
    #
    y_hat_test = y_hat_prob(sigmoid, [1.0], (0.0, 10.0), no_of_species = 1, x0 = [0.5])
    @test y_hat_test.p == [1.0]  #test object set up
    @test y_hat_test.u0 == [0.5]
    @test y_hat_test.tspan == (0.0, 10.0)
    @test y_hat_test.problem_type == DiffEqBase.StandardODEProblem()
    y_hat_test2 = y_hat_prob(sigmoid, [0.5, 1.0], (0.0, 10.0), no_of_species = 1, x0 = missing, noise = false)
    @test y_hat_test2.u0 == [0.5]
    y_hat_test3 = y_hat_prob(sigmoid, [0.5, 1.0, 0.1], (0.0, 10.0), no_of_species = 1, x0 = missing, noise = true)
    @test y_hat_test3.p == [1.0]
    y_hat_test4 = y_hat_prob(sigmoid, [1.0, 0.1], (0.0, 10.0), no_of_species = 1, x0 = [0.5], noise = true)
    @test y_hat_test4.p == [1.0]
    #
    # testing y_hat_prob() Argument Errors
    #
    @test_throws ArgumentError y_hat_prob(sigmoid, [1.0], (0.0, 10.0))
    @test_throws ArgumentError y_hat_prob(sigmoid, [1.0, -1.0], (0.0, 10.0), no_of_species = 1, x0 = [0.5], noise = true)
    @everywhere begin
        #
        #ODE arguments - setting up a dummy example y = α + βx (linear_function).
        #
        Tspan = (0.0, 100.0)  #timspan
        x0 = [0.0]  #initial conditions
        solver = RK4()
        ode_species = ["x"]
        no_of_species = length(ode_species)
        p = [1.0]
        p_σ = vcat(p, 0.1)
        number_of_pts_per_exp = Tspan[end]
        saveat = (Tspan[end] - Tspan[1])/number_of_pts_per_exp
        t_pts = collect(Tspan[1]:saveat:Tspan[end])
        #
        # user defined function that takes ODE argumetns and returns dictionary where key is "x" and value is a 2D array of size (time pts x 1)
        # real_data is given as an argument (despite not being used) as that is the specification for the user defined fn input for functions such as add_noise() and make_df()
        #
        function user_defined(real_data::DataFrame,
                              condition::String,
                              ode_species::Array{String,1},
                              ode::Function,
                              p_draw::AbstractArray{Float64,1},
                              solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                              saveat::Real;
                              x0::Union{AbstractArray{Float64,1}, Missing} = nothing,
                              maxiters::Union{Int64, Nothing} = nothing,
                              cb::Union{Nothing, Any} = nothing)

            sim_dat = get_fake_data(ode, ode_species, p_draw, saveat, t_pts, solver, x0, 1)  #number of exps = 1
            x = sim_dat[:, :, 1]
            return Dict("x" => x)
        end
        #
        #   simulating some fake data
        #
        fake_species_data = user_defined(DataFrame(), "normal", ode_species, linear_function, p_σ, solver, saveat; x0 = x0, maxiters = nothing, cb = nothing)
        noisy_data = add_noise(fake_species_data, "med", [p_σ[end]])  #just adds specified variance
        fake_data = make_df(noisy_data, t_pts, "normal")
        #
        #  define a cost function χ^2 for fitting the ODE
        #
        function cost(p)
            return residuals(linear_function, user_defined, fake_data, ode_species, p, Tspan, solver, saveat; x0 = x0, noise = true);
        end
        no_of_iters = 15    #number of experiments
        niter = 100    #no of iterations per experiment
        pyfun = so.basinhopping    #wrapper avoids an error encountered with some PyCall functions and @distributed macro
        function wrap_pyfun(cost::Function, p0::Array{Float64,1}; niter::Int64)
            return pyfun(cost, p0, niter = niter)
        end
    end
    #
    #  fit the ODE
    #
    @everywhere working_result = SharedArray{Float64,2}(length(p_σ) + 1, no_of_iters)    #julia is column major
    @sync @distributed for i in 1:no_of_iters    #fit the ODE to the fake simulated data
        p0 = vcat(rand(Normal(1, 0.1)), rand(Normal(0.1, 0.01)))
        res = wrap_pyfun(cost, convert(Array{Float64,1}, p0), niter = niter)
        working_result[:, i] = vcat(res["x"], res["fun"])
    end
    @test all(!ismissing, working_result)
    @everywhere θ_hats = working_result[1:end-1, :]    #drop the χ^2(θ) value and only keep the parameter estimates per experiment
    @everywhere begin
        θ_means = mean(θ_hats, dims = 2)
        #
        #  user defined function needed for calculation of the FIM. Takes an ODE solution & arguments needed to simulate the ODE solution
        #  and returns a dictionary where key is "x" and value is a 2D array of size (time pts x 1)
        #
        function user_defined_fim(real_data::DataFrame,
                                  condition::String,
                                  ode_species::Array{String,1},
                                  sol::T where T<:DESolution,
                                  p_draw::AbstractArray{T,1} where T<:Any,  #parameters will no longer be Floats when ODE is solved in Dual numbers
                                  solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                                  saveat::Real;
                                  x0::Union{AbstractArray{Float64,1}, Missing} = nothing,
                                  maxiters::Union{Int64, Nothing} = nothing,
                                  cb::Union{Nothing, Any} = nothing)

            sim_dat = get_fake_data(real_data, condition, ode_species, sol, p_draw, solver, saveat; x0 = x0, maxiters = maxiters, cb = cb)
            x = sim_dat[:, :, 1]
            return Dict("x" => x)
        end
    end
    #
    #  observed fisher information for a single run of the parameter estimation algorithm
    #
    F = FIM(θ_hats[:, 10],
            user_defined_fim,
            fake_data,
            linear_function,
            Tspan,
            solver,
            saveat;
            no_of_species = 1,
            x0 = x0,
            noise = true)
    y_for_test = fake_data.Reading
    #
    # check concurrance with analytic results of y = α + βx FIMs
    #
    @test isapprox(F[1, 1], sum([x^2 for x in t_pts])/θ_hats[end, 10]^2)
    @test isapprox(F[2, 2], ((-length(t_pts)/(θ_hats[end, 10]^2)) + (3/(θ_hats[end, 10]^4))*sum([(y_for_test[idx] - θ_hats[1, 10]*x).^2 for (idx, x) in enumerate(t_pts)])))
    @test isapprox(F[2, 1], (2/θ_hats[end, 10]^3)*sum([(y_for_test[idx] - θ_hats[1, 10]*x)*x for (idx, x) in enumerate(t_pts)]))
    #
    # check observed fisher information for a single run of the parameter estimation algorithmn but with no estimated noise
    #
    F_no_noise = FIM(θ_hats[1:end-1, 10],
                     user_defined_fim,
                     fake_data,
                     linear_function,
                     Tspan,
                     solver,
                     saveat;
                     no_of_species = 1,
                     x0 = x0,
                     noise = false)
    @test size(F_no_noise) == size(F) .- 1
    @test_throws ArgumentError FIM([θ_hats[1, 10]], user_defined_fim, fake_data, linear_function, Tspan, solver, saveat)
    @test ismissing(FIM([θ_hats[1, 10], -1.0], user_defined_fim, fake_data, linear_function, Tspan, solver, saveat, no_of_species = 1, noise = true))
    @everywhere ODE = TotalDEInput(linear_function, Tspan, solver, saveat, x0, 1)
    #
    #  tests alternative argument inputs for FIM()
    #
    F_alt = FIM(θ_hats[:, 10], user_defined_fim, fake_data, ODE, noise = true)
    @test F == F_alt
end
for elts in workers()  #removes the workers after testing
    rmprocs(elts);
end
