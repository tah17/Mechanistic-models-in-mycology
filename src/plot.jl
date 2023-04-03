using PyPlot; ENV["MPLBACKEND"]="tkagg"; pygui(true);
using PyCall
using Statistics
using Distributions
@pyimport pandas as pd
@pyimport seaborn as sns
sns.set(style="white", palette="muted", color_codes=true)

"""
    plot_coverage(coverages::AbstractArray{Float64,2}, binom_cis::Array{Tuple{Float64,Float64},2}, confidences::AbstractArray{Float64,1}; <keyword arguments>)

Plots the coverage against the "ideal" coverage.

#Arguments
- `coverages`: Coverage of confidence intervals for each of the simulated replicates: size of (number of parameters, number of replicates),
- `binom_cis`: Binomial confidence intervals of the coverages: size of (number of parameters, number of replicates),
- `confidences`: Confidence levels,
- `colour_palette`: Colour of the coverage plot,
- `linewidth`: Linewidth of coverage plot,
- `labelsize`: Ticker size,
- `fontsize`: Font size of labels.

#Return
- PyPlot Plot
"""
function plot_coverage(coverages::AbstractArray{Float64,2},
                       binom_cis::Array{Tuple{Float64,Float64},2},
                       confidences::AbstractArray{Float64,1};
                       colour_palette::Array{String,1} = ["#0c2c84", "red"],
                       linewidth::Int64 = 3,
                       labelsize::Int64 = 20,
                       fontsize::Int64 = 25)

    if size(coverages) != size(binom_cis)
        throw(ArgumentError("Need to provide the confidence intervals for all of the the coverage values provided."))
    end
    if size(binom_cis)[2] != length(confidences)
        throw(ArgumentError("Need to provide the same number of confidence levels as the confidence intervals were calculated for."))
    end
    lower = (x -> x[1]).(binom_cis)
    upper = (x -> x[2]).(binom_cis)
    for i in 1:size(coverages)[1]
        plt.plot(confidences, coverages[i, :], color = colour_palette[1], linewidth = linewidth)
        plt.fill_between(confidences, lower[i, :], upper[i, :], color = colour_palette[1], alpha = 0.1)
    end
    plt.plot(collect(0.0:0.1:1.0), collect(0.0:0.1:1.0), color = colour_palette[2], linestyle = "dashed", linewidth = linewidth, alpha = 0.8)
    plt.xlabel("Confidence Intervals", fontsize = fontsize)
    plt.ylabel("Coverage", fontsize = fontsize)
    plt.tick_params(labelsize = labelsize)
    return plt
end

"""
    zip_plot(θ_true::AbstractArray{Float64,1}, θ_mles::AbstractArray{Float64,2}, confit::Array{Tuple{Float64,Float64},3}, alphas = collect(0.01:0.05:0.99), alpha_idx::Int64 = 2; <keyword arguments>)

Create a zip plot for desired parameters and their corresponding confidence intervals. A zip plot stacks the calculated CIs by their bias
and standard error in order to visualise the fractional comparison of coverers (CIs including the true value) as opposed to non-coverers.
The zip plot style was introduced by (Morris et al. 2019)

#Arguments
- `θ_true`: The true parameters used in the simulation study,
- `θ_mles`: The parameter estimates from the different simulated replicates, of size (`length(θ_true)`, `n_sims`),
- `confit`: Confidence intervals from the different simulated replicates for different confidence levels, of size (`length(θ_true)`, `length(alphas)`, `n_sims`).
            These need to be same order as the `θ_mles` provided.
- `alphas`: The alpha levels that the confidence intervals were calculated for,
- `alpha_idx`: The alpha level wanted for the zip plot - defaults at ≈ 5%;
- `param_idxs`: The indices of the parameters you want to plot from `θ_true`,
- `param_labels`: The labels for the parameters which must be in the same order as the parameter indices provided,
- `labelsize`: Ticker size,
- `fontsize`:  Font size of labels.

#Return
- PyPlot Plot
"""
function zip_plot(θ_true::AbstractArray{Float64,1},
                  θ_mles::AbstractArray{Float64,2},
                  confit::Array{Tuple{Float64,Float64},3},
                  alphas = collect(0.01:0.05:0.99),
                  alpha_idx::Int64 = 2;
                  param_idxs::AbstractArray{Int64,1} = collect(1:length(θ_true)),
                  param_labels::Array{String,1} = ["\$\\theta_{$i}\$" for i in 1:length(θ_true)],
                  labelsize::Int64 = 20,
                  fontsize::Int64 = 25)

    if size(θ_mles)[2] != size(confit)[end]
        throw(ArgumentError("Need to provide the same number of replicates of parameter estimates as the confidence intervals.
        You provided $(size(θ_mles)[2]) parameter estimates but $(size(confit)[end]) confidence intervals."))
    end
    if size(θ_mles)[1] != size(confit)[1]
        throw(ArgumentError("Need to provide the same number of parameters as the confidence intervals.
        You provided $(size(θ_mles)[1]) parameters but $(size(confit)[1]) confidence intervals."))
    end
    if size(θ_mles)[1] != length(θ_true)
        throw(ArgumentError("Need to provide the same number of parameter estimates as the true parameters.
        You provided $(size(θ_mles)[1]) parameter estimates but $(length(θ_true)) true parameters."))
    end
    if size(confit)[2] != length(alphas)
        throw(ArgumentError("Need to provide the same alphas that were used to calculate the confidence intervals."))
    end
    if alpha_idx > length(alphas)
        throw(ArgumentError("Need to provide a viable alpha."))
    end
    if length(param_idxs) > length(θ_true)
        throw(ArgumentError("Parameter indices cannot exceed the number of true parameters"))
    end
    if length(param_idxs) != length(param_labels)
        throw(ArgumentError("Number of parameter labels $(length(param_labels)) do not match the number of parameters to plot $(length(param_idxs))"))
    end
    n_sims = size(θ_mles)[2]  #number of simulated replicates
    alpha = alphas[alpha_idx]  #desired alpha level to check coverage of (1 - alpha)% CIs
    no_of_params = length(param_idxs)  #number of desired paramters to plot
    θ_true_i = θ_true[param_idxs]
    θ_mles_i = θ_mles[param_idxs, :]
    confit_i = confit[param_idxs, :, :]
    quantile = abs(Statistics.quantile(Distributions.Normal(), alpha/2))
    se = ((x -> x[2]).(confit_i[:, alpha_idx, :]) .- θ_mles_i)/quantile  #standard errors of the MLEs   ## mod this before you mod the z value?
    z_vals = Array{Float64,2}(undef, no_of_params, n_sims)  #z values as seen in "Using simulation studies to evaluate statistical methods" Morris et al.
    lower_vals = Array{Float64,2}(undef, no_of_params, n_sims)
    upper_vals = Array{Float64,2}(undef, no_of_params, n_sims)
    for j in 1:no_of_params
        z_vals[j, :] = abs.([(θ_mles_i[j, i] .- θ_true_i[j])/se[j, i] for i in 1:n_sims])
        lower_vals[j, :] = (x -> x[1]).(confit_i[j, alpha_idx, sortperm(z_vals[j, :])])  #sort the CIs based on the z-values, ranks them from "good" to "bad"
        upper_vals[j, :] = (x -> x[2]).(confit_i[j, alpha_idx, sortperm(z_vals[j, :])])
    end
    cut_off = floor(n_sims*(1 - alpha)) + 0.5
    for j in 1:no_of_params
        plt.subplot("1$no_of_params$j")
        for i in 1:n_sims
            if ininterval(θ_true_i[j], (lower_vals[j, i], upper_vals[j, i]))  # different colours for coveres and non-coverers
                plt.plot([lower_vals[j, i],  upper_vals[j, i]], [i, i], color = "#9D9D9D", linewidth = 4, alpha = 0.7)
            else
                plt.plot([lower_vals[j, i],  upper_vals[j, i]], [i, i], color = "#A51900", linewidth = 4, alpha = 0.7)
            end
        end
        plt.plot(ones(n_sims)*θ_true_i[j], collect(1:n_sims), c = "#0091D4", linewidth = 3, linestyle = "dashed")
        plt.plot([minimum(lower_vals[j, :]), maximum(upper_vals[j, :])], [cut_off, cut_off], c = "#FFDD00", linewidth = 2, alpha = 0.5)
        plt.xlabel(param_labels[j], fontsize = fontsize)
        plt.tick_params(labelsize = labelsize)
        if j == 1
            plt.yticks(collect(1:23.5:n_sims), round.(collect(1:23.5:n_sims)./n_sims .*100))
        else
            plt.yticks([])
        end
    end
    return plt
end
