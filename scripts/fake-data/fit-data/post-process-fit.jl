#
#  Reads parameter estimates generated by running the corresponding file in fit-data/ .
#  Checks if all of the simulated replicates have been correctly run and stored, puts the estimates into a dataframe and plots them
#
#  e.g. if no_params = 5, algo = DE and dat_set = 5. post-process-fit.jl will read in the parameter estimates generated by running fit-data/fit-data-5.jl,
#  for parameter set 5, algo DE and all noises. Found in data/fitted-data/fake-data/fitted-data-5/DE/parameter-set-5/
#
using DelimitedFiles
using Statistics
using Random
using DataFrames
using PyPlot; ENV["MPLBACKEND"]="tkagg"; pygui(true);
using PyCall
Random.seed!(3744613754752089)
@pyimport pandas as pd
@pyimport seaborn as sns
sns.set(style = "white", palette = "muted", color_codes = true)
#
# extract the real data
#
n_sims = 100
no_params = 5
algo = "DE"
dat_set = 3
path = joinpath(@__DIR__, "../../../data")
#
# see what optimisation runs failed (if any)
#
println("dat set is $dat_set")
for j in 1:no_params
    for k in 1:5
        for i in 1:n_sims
            try
                readdlm("$path/fitted-data/fake-data/fitted-data-$dat_set/$algo/parameter-set-$j/noise-$k/working-result-$i.csv", ';', Float64)
            catch err
                println("$algo: Errors at parameter $j and noise $k at n_sims $i")
            end
        end
    end
end
#
# Collect fits into DF
#
dat_set = 1
parameter_names = ["\$F_0low\$", "\$F_0high\$", "\$C_0\$", "\$N_0\$", "\$F_d0\$", "\$\\beta\$", "\$\\delta_N\$", " \$k_NF\$", "\$k_C[M]\$", "\$\\delta_C\$", "\$\\alpha[Nv]\$", "\$k_NC\$", "\$d_MF [M]\$", "\$\\delta_F\$"]
dfs = [DataFrame() for i in 1:25]
global counter = 1
for j in 1:no_params  #cycle through parameters
    for k in 1:5  #cycle through noise
        working_result = hcat([readdlm("$path/fitted-data/fake-data/fitted-data-$dat_set/$algo/parameter-set-$j/noise-$k/working-result-$l.csv", ';', Float64) for l in 1:n_sims]...)
        df = DataFrame(working_result')
        df.noise = k
        df.parameter = j
        dfs[counter] = df
        global counter += 1
    end
end
df = join([df for df in dfs]..., kind  = :outer, on = intersect([names(df) for df in dfs]...))
#
# Simple summary plot of fits
#
parameter = 1
noise = 2
df_parameter = df[df[:parameter] .== parameter, :]
df_noise = df_parameter[df_parameter[:noise] .== noise, :]
mles = df_noise[1:14]
mc_error = [std(x)./(sqrt(2*(n_sims - 1))) for x in eachcol(mles)]
df = pd.DataFrame(convert(Array{Float64,2}, mles), columns = parameter_names)
g = sns.PairGrid(df, diag_sharey = false)
g.map_lower(sns.kdeplot, cmap = "Blues_d")
g.map_upper(plt[:scatter], edgecolor = "w", s = 40)
g.map_diag(sns.kdeplot, lw = 3)
tight_layout()