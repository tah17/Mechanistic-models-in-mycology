using HypothesisTests
using Statistics

"""
    ininterval(x::Real, interval::Tuple{Real, Real})

Checks if a value, `x`, is in a (closed) interval, `interval`.

"""
function ininterval(x::Real, interval::Tuple{T, S} where {T<:Real, S<:Real})
  if interval[1] > interval[2]
    throw(ArgumentError("Please ensure the lower value of your interval, $(interval[1]), is smaller than the upper value, $(interval[2])."))
  end
  return x >= interval[1] && x <= interval[2]
end

"""
    coverage(theta::AbstractArray{Float64,1}, confidence_intervals::Array{Tuple{Float64,Float64},3},alphas::Array{Float64,1}; level::Float64 = 0.9)

Calculates the coverage of confidence intervals, `confidence_intervals`, at confidence levels `alphas`, where `theta` is the true
parameter value and the coverage binomial confience intervals at the confidence level `level` (defaults at 10%, 5% is too conservative).

Also re-returns the confidence levels, size (`(length(alphas) + 2)`).
"""
function coverage(theta::AbstractArray{Float64,1},
                  confidence_intervals::Array{Tuple{Float64,Float64},3},
                  alphas::Array{Float64,1};
                  level::Float64 = 0.9)

  if length(theta) != size(confidence_intervals)[1]
    throw(ArgumentError("You provided $(length(theta)) true parameters but confidence intervals for $(size(confidence_intervals)[1]) parameters."))
  end
  if length(alphas) != size(confidence_intervals)[2]
    throw(ArgumentError("You provided $(length(alphas)) confidence levels but confidence intervals for $(size(confidence_intervals)[2]) levels."))
  end
  n = size(confidence_intervals)[3]
  tmp = sum(ininterval.(theta, confidence_intervals), dims = 3)
  total_coverage = reshape(tmp, size(tmp[:, :, 1]))  #work with 2D array (vs 3D)
  coverage = total_coverage./n
  binomial_tst = BinomialTest.(total_coverage, n)
  coverage_ci = confint.(binomial_tst, level = level)  #90% CIs (default is the exact clopper pearson)
  return hcat(ones(length(theta)), coverage, zeros(length(theta))), hcat([(1.0, 1.0) for x in 1:length(theta)], coverage_ci, [(0.0, 0.0) for x in 1:length(theta)]), vcat(1.0, collect(ones(length(alphas)) .- alphas), 0.0)
end
