using Plots, Plots.PlotMeasures, LaTeXStrings
using NLsolve, ForwardDiff
using DelimitedFiles
using Distributed
@everywhere using Statistics, StatsBase, Graphs, ProgressMeter, DifferentialEquations

default(legendfont = ("Computer modern", 16),
        tickfont = ("Computer modern", 16),
        guidefontsize = 18, ms = 5, fg_legend = :transparent,
        linewidth=1, framestyle=:axis, grid=:none,
        background_color_legend = AGray32(1.,0.8),
        bottom_margin = -2.5mm, left_margin = -1mm, right_margin = 1.5mm)
gr(size=(450,400))


@everywhere function tri_approx!(du,u,par,t)
    β, β△, k₁, k₁₀, k₀₁, k₁₁ = par
    I, SI, SSI₁₀, SII₁₀, SSI₀₁, SII₀₁, SSI₁₁, SII₁₁ = u

    du[1] = dI = - I + β*k₁*SI + 2*β*(k₁₀*(SSI₁₀+SII₁₀) + k₁₁*(SSI₁₁+SII₁₁)) + β△*(k₀₁*SII₀₁+k₁₁*SII₁₁)
    du[2] = dSI = - (2+β)*SI + I - ((2*SI-1+I)/(1-I))*(β*(k₁-1)*SI + 2*β*(k₁₀*(SSI₁₀+SII₁₀) + k₁₁*(SSI₁₁+SII₁₁)) + β△*(k₀₁*SII₀₁+k₁₁*SII₁₁))
    du[3] = dSSI₁₀ = - (1+2*β)*SSI₁₀ + 2*SII₁₀ - ((4*SSI₁₀-1+I+SII₁₀)/(1-I))*(β*k₁*SI + 2*β*((k₁₀-1)*(SSI₁₀+SII₁₀) + k₁₁*(SSI₁₁+SII₁₁)) + β△*(k₀₁*SII₀₁+k₁₁*SII₁₁))
    du[4] = dSII₁₀ = - (4+2*β)*SII₁₀ + I + (2*β-1)*SSI₁₀ - ((SII₁₀-2*SSI₁₀)/(1-I))*(β*k₁*SI + 2*β*((k₁₀-1)*(SSI₁₀+SII₁₀) + k₁₁*(SSI₁₁+SII₁₁)) + β△*(k₀₁*SII₀₁+k₁₁*SII₁₁))
    du[5] = dSSI₀₁ = - SSI₀₁ + 2*SII₀₁ - ((4*SSI₀₁-1+I+SII₀₁)/(1-I))*(β*k₁*SI + 2*β*(k₁₀*(SSI₁₀+SII₁₀) + k₁₁*(SSI₁₁+SII₁₁)) + β△*((k₀₁-1)*SII₀₁+k₁₁*SII₁₁))
    du[6] = dSII₀₁ = - (4+β△)*SII₀₁ + I - SSI₀₁ - ((SII₀₁-2*SSI₀₁)/(1-I))*(β*k₁*SI + 2*β*(k₁₀*(SSI₁₀+SII₁₀) + k₁₁*(SSI₁₁+SII₁₁)) + β△*((k₀₁-1)*SII₀₁+k₁₁*SII₁₁))
    du[7] = dSSI₁₁ = - (1+2*β)*SSI₁₁ + 2*SII₁₁ - ((4*SSI₁₁-1+I+SII₁₁)/(1-I))*(β*k₁*SI + 2*β*(k₁₀*(SSI₁₀+SII₁₀) + (k₁₁-1)*(SSI₁₁+SII₁₁)) + β△*(k₀₁*SII₀₁+(k₁₁-1)*SII₁₁))
    du[8] = dSII₁₁ = - (4+2*β+β△)*SII₁₁ + I + (2*β-1)*SSI₁₁ - ((SII₁₁-2*SSI₁₁)/(1-I))*(β*k₁*SI + 2*β*(k₁₀*(SSI₁₀+SII₁₀) + (k₁₁-1)*(SSI₁₁+SII₁₁)) + β△*(k₀₁*SII₀₁+(k₁₁-1)*SII₁₁))
end

@everywhere function tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, i0)
    par = [β, β△, k₁, k₁₀, k₀₁, k₁₁]
    tspan = (0.0,tmax)

    u0 = [i0, i0*(1. - i0), i0*(1. - i0)*(1. - i0), i0*i0*(1. - i0), i0*(1. - i0)*(1. - i0), i0*i0*(1. - i0), i0*(1. - i0)*(1. - i0), i0*i0*(1. - i0)]
    prob = ODEProblem(tri_approx!,u0,tspan,par)

    sol = solve(prob, Tsit5(), saveat = 1, reltol=1e-10, abstol=1e-10)
    return sol
end

function heatmap_β_vs_β△(l, κ₁, k₁, k₁₀, k₀₁, k₁₁, i0)
    β△s = LinRange(0, 4, l)
    βs = LinRange(0., 1.15, l)/(κ₁-1)

    res = zeros(l,l)
    for i in 1:l
        β = βs[i]
        for j in 1:l
            β△ = β△s[j]
            r_ = tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, i0)
            res[i,j] = r_[1,end]
        end
    end

    return (βs, β△s, res)
end

function heatmap_β_vs_k₁₁(l, β△, κ₁, κ₂, k₁, i0)
    k₁₁s = LinRange(0., minimum([κ₂,(κ₁-k₁)/2]), l)
    βs = LinRange(0.35, 1.15, l)/(κ₁-1)

    res = zeros(l,l)
    for i in 1:l
        β = βs[i]
        for j in 1:l
            k₁₁ = k₁₁s[j]
            k₁₀ = (κ₁-k₁)/2 - k₁₁
            k₀₁ = κ₂ - k₁₁
            r_ = tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, i0)
            res[i,j] = r_[1,end]
        end
    end

    return (βs, k₁₁s, res)
end

function β_cr_k₁₁only(β△, k₁₁)
    return 0.25*(β△+2)*(sqrt(1 + 8/((k₁₁-1)*(β△+2)^2)) - 1)
end

function heatmap_β△_cr(κ₁, κ₂, k₁)
    βs = LinRange(0., 1.15, 1001)/(κ₁-1)
    k₁₁s = LinRange(0., minimum([κ₂,(κ₁-k₁)/2]), 1001)

    res = zeros(1001,1001)
    for i in 1:1001
        β = βs[i]
        for j in 1:1001
            k₁₁ = k₁₁s[j]
            num = (1+2*β*(1+β))*(β*k₁*(1+2*β*(1+β)) + ((κ₁-k₁)/2)*2*β*(1+β)^2 - (1+β)*(1+2*β*(1+β)))
            den = β*((1+β)*(1+2*β*(1+β)) - β*(1+2*β*(1+β))*k₁ - 2*β*(1+β)^2*((κ₁-k₁)/2-k₁₁) - k₁₁*(1+β)*(1+2*β*(1+β)))
            res[i,j] = num/den
        end
    end

    return (βs, k₁₁s, res)
end

function f!(F,x)
    F[1] = x[1]*(k₁/(1 + x[1]) + ((κ₁-k₁)/2-k₁₁)*2*(1+x[1])/(1+2*x[1]*(1+x[1])) + k₁₁*(2*(1+x[1])+β△)/(1+2*x[1]*(1+x[1])+x[1]*β△)) - 1
end

function β_cr_vs_β△(β△s)
    global κ₁ = κ₁
    global κ₂ = κ₂
    global k₁ = k₁
    global k₁₁ = k₁₁
    sol = zeros(length(β△s))
    for i in 1:length(β△s)
        global β△ = β△s[i]
        sol[i] = nlsolve(f!, [0.3], autodiff = :forward).zero[1]
    end
    return sol
end

@everywhere begin
    i0 = 1e-6
    tmax = 5000.
    l = 101
end

κ₁ = 6
κ₂ = 3
k₁ = 0
res = heatmap_β△_cr(κ₁, κ₂, k₁)
heatmap(res[2], res[1], res[3],
    ylabel = L"\beta^{(1)}", xlabel = L"k^{(1,1)}", clims = (-0.1,7.9), xlims = [0,3],
    right_margin=5.5mm,tick_direction=:out, color=:inferno, colorbartitle = L"\beta^{(2)}",
    colorbar_titlefontsize = 18);
contour!(res[2], res[1], res[3], levs = [1,2,3,4,5,6,7], color=:white, width = 1.);
hline!([1/(κ₁-1)],label =:none,color=:white,width=1.5,ls=:dash);
hline!([β_cr_vs_β△(0.)[1]],label =:none,color=:white,width=2.)

κ₁ = 6
κ₂ = 3
k₁ = 0
β△ = 0.25
βs = LinRange(0.188,0.24,261)
res = [pmap(β -> tri_approx(β, β△, k₁, (κ₁-k₁)/2 - k₁₁, κ₂ - k₁₁, k₁₁, i0), βs) for k₁₁ in [0.,0.5,1.,1.5,2.,2.5,3.]]
plot(βs, [[res[j][i][1,end] for i in 1:length(βs)] for j in 1:7], xlabel = L"\beta^{(1)}", width = 2.5,
        labels=:none, ylabel = L"I^{\star}", palette = palette(:Blues)[[3:9;]], legend =:topleft);
vline!([1/(κ₁-1)],label =:none,color=:black,width=1.5,ls=:dash);
annotate!(0.225,0.30,text(L"k^{(1,1)}=0", "Computer modern", 15, rotation = 35));
annotate!(0.225,0.213,text(L"k^{(1,1)}=3", "Computer modern", 15, rotation = 35))
annotate!(0.214,0.35,text(L"\beta^{(2)}=0.25", "Computer modern", 15))
β△ = 1.
βs = LinRange(0.14,0.22,201)
res = [pmap(β -> tri_approx(β, β△, k₁, (κ₁-k₁)/2 - k₁₁, κ₂ - k₁₁, k₁₁, i0), βs) for k₁₁ in [0.,0.5,1.,1.5,2.,2.5,3.]]
plot(βs, [[res[j][i][1,end] for i in 1:201] for j in 1:7], xlabel = L"\beta^{(1)}", width = 2.5,
        labels=:none, ylabel = L"I^{\star}", palette = palette(:Blues)[[3:9;]], legend =:topleft);
vline!([1/(κ₁-1)],label =:none,color=:black,width=1.5,ls=:dash);
annotate!(0.18,0.678,text(L"\beta^{(2)}=1.00", "Computer modern", 15))

β△s = LinRange(0.,10., 1000)
plot(β△s, [β_cr_k₁₁only.(β△s, k₁₁) for k₁₁ in [2,3,4,5]], yscale=:lin, palette=palette(:Reds)[[3;5;7;9]],
    xlabel = L"\beta^{(2)}", ylabel = L"\beta^{(1)}_\textrm{cr}", width = 3., grid=:none, left_margin = 0.2mm,
    labels = [" 2" " 3" " 4" " 5"], legend_title = L"k^{(1,1)}", legend_title_font_pointsize = 16)


#########  Montecarlo  #########

include("aux.jl")

begin
    max_timesteps = 200000
    Δt = 0.002
    average_window = 20000
    M = 50
    update_p = 0.025
    n = 12
end

### Simplicial Complex ###
N = 5000
k₁, k₁₀, k₀₁, k₁₁ = 2, 0, 0, 3

results_MC_02 = readdlm("/results/random_regular/MC_betaTri_02.txt")
results_MC_06 = readdlm("/results/random_regular/MC_betaTri_06.txt")
results_MC_1 = readdlm("/results/random_regular/MC_betaTri_1.txt")
results_MC_1_b = readdlm("/results/random_regular/MC_betaTri_1_back.txt")

βs = LinRange(0.08,0.16,201)
results_MF = [map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 1/N)[1,end], βs) for β△ in [0.2,0.6,1.]]
results_MF_b = [map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 0.8)[1,end], βs) for β△ in [0.2,0.6,1.]]

βs = LinRange(0.115,0.16,46)
scatter(βs, results_MC_02[:,1], yerr = results_MC_02[:,2], legend=:topleft,
        label = " 0.2", legend_title = L"\beta^{(2)}", legend_title_font_pointsize = 16, ms = 7, ylims = [-0.025,0.72],
        xlabel = L"\beta^{(1)}", ylabel = L"I^\star", palette=palette(:Blues)[[5]]);
βs = LinRange(0.08,0.16,201)
plot!(βs, results_MF[1], width = 3.5, palette=palette(:Blues)[[5]], label =:none);
βs = LinRange(0.11,0.16,51)
scatter!(βs, results_MC_06[:,1], yerr = results_MC_06[:,2],
        label = " 0.6", palette=palette(:Blues)[[7]], ms = 7);
βs = LinRange(0.08,0.16,201)
plot!(βs, results_MF[2], width = 3.5, palette=palette(:Blues)[[7]], label =:none);
βs = LinRange(0.08,0.16,81)
scatter!(βs, results_MC_1[:,1], yerr = results_MC_1[:,2],
        label = " 1.0", palette=palette(:Blues)[[9]], ms = 7);
βs = LinRange(0.08,0.112,33)
scatter!(βs, results_MC_1_b[:,1], yerr = results_MC_1_b[:,2],
        label =:none, palette=palette(:Blues)[[9]], ms = 7);
βs = LinRange(0.08,0.16,201)
plot!(βs, results_MF[3], width = 3.5, palette=palette(:Blues)[[9]], label =:none);
plot!(βs, results_MF_b[3], width = 3.5, palette=palette(:Blues)[[9]], label =:none)


### Simple Hypergraph ###
N = 5000
k₁, k₁₀, k₀₁, k₁₁ = 2, 3, 3, 0

results_MC_02_SH = readdlm("/results/random_regular/MC_betaTri_02_SH.txt")
results_MC_06_SH = readdlm("/results/random_regular/MC_betaTri_06_SH.txt")
results_MC_06_b_SH = readdlm("/results/random_regular/MC_betaTri_06_back_SH.txt")
results_MC_1_SH = readdlm("/results/random_regular/MC_betaTri_1_SH.txt")
results_MC_1_b_SH = readdlm("/results/random_regular/MC_betaTri_1_back_SH.txt")

βs = LinRange(0.08,0.16,201)
results_MF = [map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 1/N)[1,end], βs) for β△ in [0.2,0.6,1.]]
results_MF_b = [map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 0.8)[1,end], βs) for β△ in [0.2,0.6,1.]]

βs = LinRange(0.125,0.16,36)
scatter(βs, results_MC_02_SH[:,1], yerr = results_MC_02_SH[:,2], legend=:topleft,
        label = " 0.2", legend_title = L"\beta^{(2)}", legend_title_font_pointsize = 16, ms = 7, ylims = [-0.025,0.72],
        xlabel = L"\beta^{(1)}", ylabel = L"I^\star", palette=palette(:Blues)[[5]]);
βs = LinRange(0.08,0.16,201)
plot!(βs, results_MF[1], width = 3.5, palette=palette(:Blues)[[5]], label =:none);
βs = LinRange(0.125,0.16,36)
scatter!(βs, results_MC_06_SH[:,1], yerr = results_MC_06_SH[:,2],
        label = " 0.6", palette=palette(:Blues)[[7]], ms = 7);
βs = LinRange(0.11,0.146,37)
scatter!(βs, results_MC_06_b_SH[:,1], yerr = results_MC_06_b_SH[:,2],
        label =:none, palette=palette(:Blues)[[7]], ms = 7);
βs = LinRange(0.08,0.16,201)
plot!(βs, results_MF[2], width = 3.5, palette=palette(:Blues)[[7]], label =:none);
plot!(βs, results_MF_b[2], width = 3.5, palette=palette(:Blues)[[7]], label =:none);
βs = LinRange(0.08,0.16,81)
scatter!(βs, results_MC_1_SH[:,1], yerr = results_MC_1_SH[:,2],
        label = " 1.0", palette=palette(:Blues)[[9]], ms = 7);
βs = LinRange(0.08,0.142,63)
scatter!(βs, results_MC_1_b_SH[:,1], yerr = results_MC_1_b_SH[:,2],
        label =:none, palette=palette(:Blues)[[9]], ms = 7);
βs = LinRange(0.08,0.16,201)
plot!(βs, results_MF[3], width = 3.5, palette=palette(:Blues)[[9]], label =:none);
plot!(βs, results_MF_b[3], width = 3.5, palette=palette(:Blues)[[9]], label =:none, legend=:none)
