#
#  Reads profile confidence intervals generated by running the corresponding file in profile-likelihood/ (in this directory) and plots the
#  confidence interval coverages, along with zip plots.
#
#  e.g. if p_set = 2, chosen_noise = 3 and dat_set = 1. param-fit-vis.jl will read in the bootstrap estimates egnerated by running profile-likelihood/profile-1.jl,
#  for parameter set 2 and noise 3. Found in data/profile-intervals/profile-intervals-1/DE/parameter-set-2/noise-3/
#
using DelimitedFiles

include("../../../src/coverage.jl")
include("../../../src/plot.jl")
include("../../../src/bootstrap.jl")
path = joinpath(@__DIR__, "../../../data")

p_set = 2  #pick a parameter set from 1 to 5
chosen_noise = 2  #pick a noise level
dat_set = 5  #pick an experimental design
tmp = readdlm("$path/parameters.csv", ';')[p_set, :]
if dat_set == 2
    p_true = convert(Array{Float64,1},  vcat(tmp[1], tmp[3:end]))
else
    p_true = convert(Array{Float64,1}, tmp)
end
n_sims = 100
alphas = collect(0.01:0.05:0.99)
θ_mles = Array{Union{Float64, Missing},2}(undef, length(p_true), n_sims)
for i in 1:n_sims
    try
        θ_mles[:, i] = readdlm("$path/fitted-data/fake-data/fitted-data-$dat_set/DE/parameter-set-$p_set/noise-$chosen_noise/working-result-$i.csv", ';', Float64)[1:end-1]
    catch err
        θ_mles[:, i] = Array{Missing,1}(missing, length(p_true))
    end
end
θ_mles_no_missing = reshape(collect(skipmissing(θ_mles)), length(p_true), n_sims - length(findall(ismissing, θ_mles[1, :])))
confit = Array{Union{Tuple{Float64,Float64}, Missing},3}(undef, length(p_true), length(alphas), n_sims)
for k in 1:n_sims
    try
        tmp = readdlm("$path/profile-intervals/profile-intervals-$dat_set/DE/parameter-set-$p_set/noise-$chosen_noise/profile-intervals-$k.csv", ';', Any)
        confit[:, :, k] = [Tuple(parse.(Float64, split(x, ['(', ',', ')'])[2:end-1])) for x in tmp]
    catch err
        confit[:, :, k] = Array{Missing,2}(missing, length(p_true),  length(alphas))
    end
end
confit_no_missing = reshape(collect(skipmissing(confit)), length(p_true), length(alphas), n_sims - length(findall(ismissing, confit[1, 1, :])))
θ_mles_confit = convert(Array{Float64,2}, θ_mles[:, findall(!ismissing, confit[1, 1, :])])
#
# calculate coverage
#
for i in 1:(n_sims - length(findall(ismissing, confit[1, 1, :])))
    for j in 1:length(alphas)
        for k in 1:length(p_true)
            if confit_no_missing[k, j, i][2] < 1e-12
                confit_no_missing[k, j, i] = tuple(0, 4)
            elseif  confit_no_missing[k, j, i][2] < confit_no_missing[k, j, i][1]
                confit_no_missing[k, j, i] = tuple(0, 4)
            end
        end
    end
end
coverages, binom_cis, confidences = coverage(p_true, confit_no_missing, alphas)
#
# plot coverage
#
plt.figure(figsize = (5, 5))
plot_coverage(coverages, binom_cis, confidences)
tight_layout()
savefig("coverage_pl_data_set_$dat_set.pdf", transparent = true)
#
# zip plots
#
p_idxs = collect(length(p_true)-8:length(p_true))
p_label = ["\$\\theta_{$i}\$" for i in p_idxs]
plt.figure(figsize = (4, 8))
zip_plot(p_true, θ_mles_confit, confit_no_missing, param_idxs = p_idxs, param_labels = p_label)
tight_layout()
# savefig("data_set_$dat_set.pdf", transparent = true)
