#
#  READS IN THE REAL DATA THAT ARE STORED IN SEPERATE .XLSX SPREADSHEETS INTO A SINGLE DATAFRAME
#
using DelimitedFiles
using DataFrames
using Random
using CSV
Random.seed!(3744613754752089)
include("../../src/functions.jl")
using PyPlot; ENV["MPLBACKEND"]="tkagg"; pygui(true);
#
# extract the real data
#
filepath = joinpath(@__DIR__, "../..")
xf_il8 = XLSX.readxlsx("/$filepath/data/IL-8.xlsx")["Sheet1"][:]
wanted_idxs_il8 = [1:10, 13:16]  #wanted columns of excel spreadsheet
xf_n = XLSX.readxlsx("/$filepath/data/N data.xlsx")["Sheet1"][:]
wanted_idxs_n = [1:9, 12:15]
xf_tnfalpha = XLSX.readxlsx("/$filepath/data/TNF alpha.xlsx")["Sheet1"][:]
wanted_idxs_tnfalpha = [1:16, 19:25]
xf_f = XLSX.readxlsx("/$filepath/data/F_all.xlsx")["Sheet1"][:]
wanted_idxs_f_all = [1:17]
xf_f_low = XLSX.readxlsx("/$filepath/data/F_all.xlsx")["Sheet1"][:]
wanted_idxs_f_low = [19:38]
xf_tot = [xf_f, xf_f_low, xf_tnfalpha, xf_il8, xf_n]  #list of all excel sheets
wanted_idxs_tot = [wanted_idxs_f_all, wanted_idxs_f_low, wanted_idxs_tnfalpha, wanted_idxs_il8, wanted_idxs_n]  #columns
start_of_data_tot = [2, 2, 19, 19, 2]  #rows where the data begins in the excel spreadsheets
raw_data = get_data(xf_tot, wanted_idxs_tot, start_of_data_tot)
raw_data[:, 2:end, :] = raw_data[:, 2:end, :]./(10^3)  #converting to the correct units for ODE model
raw_data[:, 2:end, 1:2] = raw_data[:, 2:end, 1:2]./(10^3)  #converting to the correct units for ODE model
t_pts = convert(Array{Float64,1}, raw_data[:, 1, 1])  #get the unique timepoints
no_exps = [length(vcat(wanted_idxs_tot[i]...)) - 1 for i in 1:length(wanted_idxs_tot)]  #number of experiments in the spreadsheet
#
# put the data into dataframes
#
df_F = DataFrame(Time = repeat(t_pts, outer = no_exps[1]),
                 Reading = reshape(raw_data[:, 2:no_exps[1] + 1, 1], no_exps[1]*length(t_pts)),
                 Experiment = repeat(["E$i" for i in 1:no_exps[1]], inner = length(t_pts)),
                 Species = repeat(["F_all"], inner = length(t_pts)*no_exps[1]),
                 Condition = repeat(["High IC"], inner = length(t_pts)*no_exps[1]))
df_F_low = DataFrame(Time = repeat(t_pts, outer = no_exps[2]),
                     Reading = reshape(raw_data[:, 2:no_exps[2] + 1, 2], no_exps[2]*length(t_pts)),
                     Experiment = repeat(["E$i" for i in no_exps[1] + 1:no_exps[1] + no_exps[2]], inner = length(t_pts)),
                     Species = repeat(["F_all"], inner = length(t_pts)*no_exps[2]),
                     Condition = repeat(["Low IC"], inner = length(t_pts)*no_exps[2]))
df_C = DataFrame(Time = repeat(t_pts, outer = (no_exps[3] + no_exps[4])),
                 Reading = vcat(reshape(raw_data[:, 2:no_exps[3] + 1, 3], no_exps[3]*length(t_pts)), reshape(raw_data[:, 2:no_exps[4] + 1, 4], no_exps[4]*length(t_pts))),
                 Experiment = repeat(["E$i" for i in sum(no_exps[1:2]) + 1:sum(no_exps[1:4])], inner = length(t_pts)),
                 Species = repeat(["C"], inner = (no_exps[3] + no_exps[4])*length(t_pts)),
                 Condition = repeat(["High IC"], inner = (no_exps[3] + no_exps[4])*length(t_pts)))
df_N = DataFrame(Time = repeat(t_pts, outer = no_exps[5]),
                 Reading = reshape(raw_data[:, 2:no_exps[5] + 1, 5], no_exps[5]*length(t_pts)),
                 Experiment = repeat(["E$i" for i in sum(no_exps[1:4]) + 1:sum(no_exps[1:5])], inner = length(t_pts)),
                 Species = repeat(["N"], inner = length(t_pts)*no_exps[5]),
                 Condition = repeat(["High IC"], inner = length(t_pts)*no_exps[5]))
df = join(df_F, df_F_low, df_C, df_N, kind  = :outer, on = intersect(names(df_F), names(df_F_low), names(df_C), names(df_N)))
df_innate_data = filter(x -> x.Time <= 80, df)  #we are only interested in dynamics within the first 80 hours
data = filter(x -> !ismissing(x.Reading), df_innate_data)
CSV.write("data.csv", data)
#
# plots real data
#
data_high_ic = filter(x -> x.Condition == "High IC", data)
colours = ["darkgreen", "#fc8d62", "navy"]
limits = [(10^4, 10^8), (10^1, 10^4), (10^3, 10^8)]
for (idx, species_data) in enumerate(groupby(data_high_ic, :Species))
    plt.subplot("13$idx")
    if idx == 2
        plt.scatter(species_data.Time, species_data.Reading.*10^3, marker = "o", color = colours[idx], linewidth = 2, alpha = 0.6)
    else
        plt.scatter(species_data.Time, species_data.Reading.*10^6, marker = "x", color = colours[idx], linewidth = 2, alpha = 0.6)
    end
    plt.xlabel("Time (hrs)", fontsize = 20)
    plt.tick_params(labelsize = 20)
    plt.yscale("log")
    plt.ylim(limits[idx])
end
tight_layout()
savefig("real_data.pdf", transparent = true)
