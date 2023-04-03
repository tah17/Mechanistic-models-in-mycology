using Test
# Julia Version 1.3.1
# Package dependencies for testing:
# Atom v0.12.21
# BenchmarkTools v0.5.0
# CSV v0.8.4
# DataFrames v0.20.2
# DifferentialEquations v6.11.0
# Distributions v0.22.4
# ForwardDiff v0.10.9
# HypothesisTests v0.9.1
# IJulia v1.21.1
# Juno v0.8.0
# LikelihoodProfiler v0.2.0
# Optim v0.20.1
# PyCall v1.91.3
# PyPlot v2.8.2
# ScikitLearn v0.6.2
# Sundials v3.9.0
# SymPy v1.0.17
# TimerOutputs v0.5.3
# XLSX v0.6.0
# Zygote v0.5.3

@testset "Mechanistic models in mycology" begin
    include("data-handling.jl")
    include("functions.jl")
    @testset "Confidence Intervals" begin
            include("theoretical.jl")
            include("bootstrap.jl")
    end
    include("coverage.jl")
    include("misc.jl")
    include("fit.jl")
end
