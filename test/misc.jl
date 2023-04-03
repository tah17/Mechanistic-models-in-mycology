using Test
using LinearAlgebra

include("../src/misc.jl")

@testset "Miscellaneous Functions" begin
    @test operation_on_missing(diag, ones(5,5)) == ones(5)
    @test operation_on_missing.(sqrt, ones(5,5)*25) == 5.0.*ones(5,5)
    missing_arr = Array{Union{Missing, Float64},2}(missing, (5,5))
    @test ismissing(operation_on_missing(sqrt, missing_arr))
    @test ismissing(operation_on_missing(sqrt, missing))
    @test operation_on_missing(sqrt, 25) == 5.0
    missing_arr[1,:] = ones(5)*25.0
    @test isequal(operation_on_missing.(sqrt, missing_arr), sqrt.(missing_arr))
    @test isequal(operation_on_missing(diag, missing_arr), diag(missing_arr))
    @test_throws MethodError inv(missing_arr)
    @test ismissing(operation_on_missing(inv, missing_arr))
end
