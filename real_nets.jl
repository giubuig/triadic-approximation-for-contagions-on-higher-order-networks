using SparseArrays, Graphs, DelimitedFiles

function get_triangles(G, neigh)
    tri = triangles(G)

    tris = zeros(Int64, Int(sum(tri)/3), 3)
    num_tri = 0
    for i in 1:nv(G)
        if tri[i] != 0
            for j in neigh[i]
                if j > i
                    for l in neigh[j]
                        if l > j
                            if i in neigh[l]
                                num_tri += 1
                                tris[num_tri, :] .= [i, j, l]
                            end
                        end
                    end
                end
            end
        end
    end
    return tris
end

function AggregateTemporalEdgeList(temporal_links)
    N = length(unique(temporal_links[:,2:3]))
    L = size(temporal_links, 1)
    W = sparse(zeros(Int64,N,N))

    for l in 1:L
        i = Int64(temporal_links[l,2])
        j = Int64(temporal_links[l,3])
        W[i,j] += 1
        W[j,i] += 1
    end

    WL = zeros(Int64, 0, 3)
    for i in 1:N
        for j in (i+1):N
            w = W[i,j]
            if w > 0
                WL = vcat(WL, [i, j, w]')
            end
        end
    end

    return(W, WL)
end

function graph_from_weighted_edgelist(N, edgelist)
    G = SimpleGraph(N)
    adj_mtx = sparse(zeros(Int8,N,N))
    edgelist = edgelist[sortperm(edgelist[:, 3], rev = true), :]
    l = 0
    while count(x -> x == 0, degree(G)) > 0.05*N
        l += 1
        i, j = edgelist[l,1:2]
        add_edge!(G,i,j)
        adj_mtx[i,j] = 1
        adj_mtx[j,i] = 1
        if maximum(length.(maximal_cliques(G))) > 3  # avoid cliques with more than 3 nodes
            rem_edge!(G,i,j)
            adj_mtx[i,j] = 0
            adj_mtx[j,i] = 0
        end
    end
    w_filt = edgelist[l,3]
    return G, adj_mtx, w_filt
end

function connect_graph!(G, adj_mtx)
    isol_nodes = findall(x -> x == 0, degree(G))
    conn_nodes = findall(x -> x > 0, degree(G))
    for i in 1:length(isol_nodes)
        n1 = isol_nodes[i]
        n2 = sample(conn_nodes)
        add_edge!(G,n1,n2)
        adj_mtx[n1,n2] = 1
        adj_mtx[n2,n1] = 1
    end
    cc = connected_components(G)
    for i in 1:(length(cc)-1)
        n1 = sample(cc[i])
        n2 = sample(cc[i+1])
        add_edge!(G,n1,n2)
        adj_mtx[n1,n2] = 1
        adj_mtx[n2,n1] = 1
    end
end

function rank_3_complex_from_real_net(G, adj_mtx, w_filt, cycles3, h)
    N = nv(G)
    Pairs = Vector{Array{Int64,1}}()
    Triangles = Vector{Array{Int64,1}}()
    n_cycles3 = size(cycles3,1)
    adj_tri_mtx = sparse(zeros(Int8,N,N))

    for _ in 1:n_cycles3
        if rand() < h
            t = sample([1:n_cycles3;])
            while cycles3[t,:] ∈ Triangles
                t = sample([1:n_cycles3;])
            end
            push!(Triangles, cycles3[t,:])
        else
            i, j, l = sort!(sample([1:N;],3))
            while (adj_mtx[i,j] > 0 || adj_mtx[i,l] > 0 || adj_mtx[j,l] > 0)
                i, j, l = sort!(sample([1:N;],3,replace=false))
            end
            push!(Triangles, [i,j,l])
        end
    end

    for t in 1:n_cycles3
        i, j, k = cycles3[t,:]
        adj_tri_mtx[i,j] += 1
        adj_tri_mtx[j,i] += 1
        adj_tri_mtx[i,k] += 1
        adj_tri_mtx[k,i] += 1
        adj_tri_mtx[j,k] += 1
        adj_tri_mtx[k,j] += 1
    end
    
    L = 0
    for i in 1:N
        for j in i:N
            if adj_tri_mtx[i,j] > 0
                [push!(Pairs, [i,j]) for _ in 1:adj_tri_mtx[i,j]]
            elseif adj_tri_mtx[i,j] == 0 && adj_mtx[i,j] > 0
                push!(Pairs, [i,j])
                L += 1
            end
        end
    end
    
    return Pairs, Triangles, 2*L/N
end


### University ###

# temporal_links = DelimitedFiles.readdlm("real_nets/t_edges_dbm74.dat", '\t')
# weights_mtx, w_links_list = AggregateTemporalEdgeList(temporal_links)
# N = size(weights_mtx,1)
# G, adj_mtx, w_filt = graph_from_weighted_edgelist(N, w_links_list)
# connect_graph!(G, adj_mtx)
# neigh = [neighbors(G,i) for i in 1:N]
# tri = get_triangles(G, neigh)

# Pairs_h_0, Triangles_h_0, k₁ = rank_3_complex_from_real_net(G, adj_mtx, w_filt, tri, 0.)
# Pairs_h_05, Triangles_h_05, k₁ = rank_3_complex_from_real_net(G, adj_mtx, w_filt, tri, 0.5)
# Pairs_h_1, Triangles_h_1, k₁ = rank_3_complex_from_real_net(G, adj_mtx, w_filt, tri, 1.)

# βs = [0.001:0.001:0.09;]
# β△ = 0.3
# results_MC_univ_betaTri_03_h_0, results_MC_univ_betaTri_03_h_0_det = get_results_MC_real_net(N, Pairs_h_0, Triangles_h_0, βs, β△, 1., 1/N, max_timesteps, Δt, average_window, M, update_p, n)
# results_MC_univ_betaTri_03_h_05, results_MC_univ_betaTri_03_h_05_det = get_results_MC_real_net(N, Pairs_h_05, Triangles_h_05, βs, β△, 1., 1/N, max_timesteps, Δt, average_window, M, update_p, n)
# results_MC_univ_betaTri_03_h_1, results_MC_univ_betaTri_03_h_1_det = get_results_MC_real_net(N, Pairs_h_1, Triangles_h_1, βs, β△, 1., 1/N, max_timesteps, Δt, average_window, M, update_p, n)

# results_MC_univ_betaTri_03_h_0_ = zeros(length(βs),3)
# results_MC_univ_betaTri_03_h_05_ = zeros(length(βs),3)
# results_MC_univ_betaTri_03_h_1_ = zeros(length(βs),3)
# for i in 1:length(βs)
#     results_MC_univ_betaTri_03_h_0_[i,:] .= quantile(results_MC_univ_betaTri_03_h_0_det[i,:],0.5), quantile(results_MC_univ_betaTri_03_h_0_det[i,:],0.05), quantile(results_MC_univ_betaTri_03_h_0_det[i,:],0.95)
#     results_MC_univ_betaTri_03_h_05_[i,:] .= quantile(results_MC_univ_betaTri_03_h_05_det[i,:],0.5), quantile(results_MC_univ_betaTri_03_h_05_det[i,:],0.05), quantile(results_MC_univ_betaTri_03_h_05_det[i,:],0.95)
#     results_MC_univ_betaTri_03_h_1_[i,:] .= quantile(results_MC_univ_betaTri_03_h_1_det[i,:],0.5), quantile(results_MC_univ_betaTri_03_h_1_det[i,:],0.05), quantile(results_MC_univ_betaTri_03_h_1_det[i,:],0.95)
# end

βs = [0.001:0.001:0.09;]
β△ = 0.3
results_MC_univ_betaTri_03_h_0_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_03_h_0.txt")
results_MC_univ_betaTri_03_h_05_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_03_h_05.txt")
results_MC_univ_betaTri_03_h_1_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_03_h_1.txt")

scatter(βs,results_MC_univ_betaTri_03_h_0_[1:90,1], width=0., legend =:bottomright,
        label=L"0.0", ms = 7, palette=palette(:Blues)[[3]]);
plot!(βs,results_MC_univ_betaTri_03_h_0_[1:90,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_03_h_0_[1:90,1] .- results_MC_univ_betaTri_03_h_0_[1:90,2], results_MC_univ_betaTri_03_h_0_[1:90,3] .- results_MC_univ_betaTri_03_h_0_[1:90,1]));
scatter!(βs,results_MC_univ_betaTri_03_h_05_[1:90,1], width=0.,
        label=L"0.5", ms = 7, palette=palette(:Blues)[[6]]);
plot!(βs,results_MC_univ_betaTri_03_h_05_[1:90,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_03_h_05_[1:90,1] .- results_MC_univ_betaTri_03_h_05_[1:90,2], results_MC_univ_betaTri_03_h_05_[1:90,3] .- results_MC_univ_betaTri_03_h_05_[1:90,1]));
scatter!(βs,results_MC_univ_betaTri_03_h_1_[1:90,1], width=0.,
        label=L"1.0", title = L"\beta^{(2)}=%$β△", titlefontsize = 16, xlabel = L"\beta^{(1)}", ylabel = L"I^\star",
        legend_title = L"h", legend_title_font_pointsize = 16, ms = 7, palette=palette(:Blues)[[9]]);
plot!(βs,results_MC_univ_betaTri_03_h_1_[1:90,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_03_h_1_[1:90,1] .- results_MC_univ_betaTri_03_h_1_[1:90,2], results_MC_univ_betaTri_03_h_1_[1:90,3] .- results_MC_univ_betaTri_03_h_1_[1:90,1]),
        ylims = [-0.02,0.63])

βs = [0.001:0.0002:0.09;]
κ₁ = 2*size(Pairs_h_0,1)/N
κ₂ = 3*size(Triangles_h_0,1)/N
h = 1
k₁₁ = h*κ₂
k₀₁ = (1-h)*κ₂
k₁₀ = (κ₁ - k₁)/2 - k₁₁
results_MF = map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 1/N)[1,end], βs)
plot!(βs, results_MF, width = 4, palette=palette(:Blues)[[9]], label =:none, ylims = [-0.02,0.65])


βs = [0.001:0.001:0.1;]
β△ = 0.5
results_MC_univ_betaTri_05_h_0_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_05_h_0.txt")
results_MC_univ_betaTri_05_h_05_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_05_h_05.txt")
results_MC_univ_betaTri_05_h_1_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_05_h_1.txt")

scatter(βs[1:90],results_MC_univ_betaTri_05_h_0_[1:90,1], width=0., legend =:bottomright,
        label=L"0.0", ms = 7, palette=palette(:Blues)[[3]]);
plot!(βs[1:90],results_MC_univ_betaTri_05_h_0_[1:90,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_05_h_0_[1:90,1] .- results_MC_univ_betaTri_05_h_0_[1:90,2], results_MC_univ_betaTri_05_h_0_[1:90,3] .- results_MC_univ_betaTri_05_h_0_[1:90,1]));
scatter!(βs[1:90],results_MC_univ_betaTri_05_h_05_[1:90,1], width=0.,
        label=L"0.5", ms = 7, palette=palette(:Blues)[[6]]);
plot!(βs[1:90],results_MC_univ_betaTri_05_h_05_[1:90,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_05_h_05_[1:90,1] .- results_MC_univ_betaTri_05_h_05_[1:90,2], results_MC_univ_betaTri_05_h_05_[1:90,3] .- results_MC_univ_betaTri_05_h_05_[1:90,1]));
scatter!(βs[1:90],results_MC_univ_betaTri_05_h_1_[1:90,1], width=0.,
        label=L"1.0", title = L"\beta^{(2)}=%$β△", titlefontsize = 16, xlabel = L"\beta^{(1)}", ylabel = L"I^\star",
        legend_title = L"h", legend_title_font_pointsize = 16, ms = 7, palette=palette(:Blues)[[9]]);
plot!(βs[1:90],results_MC_univ_betaTri_05_h_1_[1:90,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_05_h_1_[1:90,1] .- results_MC_univ_betaTri_05_h_1_[1:90,2], results_MC_univ_betaTri_05_h_1_[1:90,3] .- results_MC_univ_betaTri_05_h_1_[1:90,1]),
        ylims = [-0.025,0.74])

βs = [0.001:0.0002:0.09;]
κ₁ = 2*size(Pairs_h_0,1)/N
κ₂ = 3*size(Triangles_h_0,1)/N
h = 1
k₁₁ = h*κ₂
k₀₁ = (1-h)*κ₂
k₁₀ = (κ₁ - k₁)/2 - k₁₁
results_MF = map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 1/N)[1,end], βs)
plot!(βs, results_MF, width = 4, palette=palette(:Blues)[[9]], label =:none, ylims = [-0.02,0.77])

# SIR
results_MC_univ_betaTri_0_h_0_SIR_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_0_h_0_SIR.txt")
results_MC_univ_betaTri_2_h_0_SIR_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_2_h_0_SIR.txt")
results_MC_univ_betaTri_2_h_1_SIR_ = DelimitedFiles.readdlm("/results/university/results_MC_univ_betaTri_2_h_1_SIR.txt")

βs = 10 .^LinRange(-3,-0.5,26)
default(legendfont = ("Computer modern", 16),
        tickfont = ("Computer modern", 16),
        guidefontsize = 18, ms = 5, fg_legend = :transparent,
        linewidth=1, framestyle=:axis, grid=:none,
        background_color_legend = AGray32(1.,0.8),
        bottom_margin = 1.5mm, left_margin = 1mm, right_margin = 1.5mm)
# gr(size=(550,390))  # for lin plot
gr(size=(550,400))  # for log plot
scatter(βs,results_MC_univ_betaTri_0_h_0_SIR_[:,1], width=0.,
        label=L"β^{(2)} = 0.0,\ \forall\ h", ms = 7, palette=palette(:Greys)[[6]]);
plot!(βs,results_MC_univ_betaTri_0_h_0_SIR_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_0_h_0_SIR_[:,1] .- results_MC_univ_betaTri_0_h_0_SIR_[:,2], results_MC_univ_betaTri_0_h_0_SIR_[:,3] .- results_MC_univ_betaTri_0_h_0_SIR_[:,1]));
scatter!(βs,results_MC_univ_betaTri_2_h_0_SIR_[:,1], width=0.,
        label=L"β^{(2)} = 2.0,\ h = 0", ms = 7, palette=palette(:Blues)[[6]]);
plot!(βs,results_MC_univ_betaTri_2_h_0_SIR_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_2_h_0_SIR_[:,1] .- results_MC_univ_betaTri_2_h_0_SIR_[:,2], results_MC_univ_betaTri_2_h_0_SIR_[:,3] .- results_MC_univ_betaTri_2_h_0_SIR_[:,1]));
scatter!(βs,results_MC_univ_betaTri_2_h_1_SIR_[:,1], width=0.,# legend=:bottomright,
        label=L"β^{(2)} = 2.0,\ h = 1", titlefontsize = 16, xlabel = L"\beta^{(1)}", ylabel = L"R_\infty",
        ms = 7, palette=palette(:Reds)[[6]], legend=:none, xscale=:log10,
        );
plot!(βs,results_MC_univ_betaTri_2_h_1_SIR_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_univ_betaTri_2_h_1_SIR_[:,1] .- results_MC_univ_betaTri_2_h_1_SIR_[:,2], results_MC_univ_betaTri_2_h_1_SIR_[:,3] .- results_MC_univ_betaTri_2_h_1_SIR_[:,1]))



### Conference ###

# temporal_links = DelimitedFiles.readdlm("/real_nets/tij_pres_InVS15.dat", '\t')
# weights_mtx, w_links_list = AggregateTemporalEdgeList(temporal_links)
# N = size(weights_mtx,1)
# G, adj_mtx, w_filt = graph_from_weighted_edgelist(N, w_links_list)
# connect_graph!(G, adj_mtx)
# neigh = [neighbors(G,i) for i in 1:N]
# tri = get_triangles(G, neigh)

βs = [0.001:0.001:0.09;]
β△ = 0.08
results_MC_conf_betaTri_008_h_0_ = DelimitedFiles.readdlm("/results/conference/results_MC_conf_betaTri_008_h_0.txt")
results_MC_conf_betaTri_008_h_05_ = DelimitedFiles.readdlm("/results/conference/results_MC_conf_betaTri_008_h_05.txt")
results_MC_conf_betaTri_008_h_1_ = DelimitedFiles.readdlm("/results/conference/results_MC_conf_betaTri_008_h_1.txt")

scatter(βs,results_MC_conf_betaTri_008_h_0_[:,1], width=0., ylims = [-0.025,0.9], legend =:bottomright,
        label=L"0.0", ms = 7, palette=palette(:Blues)[[3]]);
plot!(βs,results_MC_conf_betaTri_008_h_0_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_conf_betaTri_008_h_0_[:,1] .- results_MC_conf_betaTri_008_h_0_[:,2], results_MC_conf_betaTri_008_h_0_[:,3] .- results_MC_conf_betaTri_008_h_0_[:,1]));
scatter!(βs,results_MC_conf_betaTri_008_h_05_[:,1], width=0.,
        label=L"0.5", ms = 7, palette=palette(:Blues)[[6]]);
plot!(βs,results_MC_conf_betaTri_008_h_05_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_conf_betaTri_008_h_05_[:,1] .- results_MC_conf_betaTri_008_h_05_[:,2], results_MC_conf_betaTri_008_h_05_[:,3] .- results_MC_conf_betaTri_008_h_05_[:,1]));
scatter!(βs,results_MC_conf_betaTri_008_h_1_[:,1], width=0.,
        label=L"1.0", title = L"\beta^{(2)}=%$β△", titlefontsize = 16, xlabel = L"\beta^{(1)}", ylabel = L"I^\star",
        legend_title = L"h", legend_title_font_pointsize = 16, ms = 7, palette=palette(:Blues)[[9]]);
plot!(βs,results_MC_conf_betaTri_008_h_1_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_conf_betaTri_008_h_1_[:,1] .- results_MC_conf_betaTri_008_h_1_[:,2], results_MC_conf_betaTri_008_h_1_[:,3] .- results_MC_conf_betaTri_008_h_1_[:,1]),
        ylims = [-0.02,0.7])

κ₁ = 2*size(Pairs_h_0,1)/N
κ₂ = 3*size(Triangles_h_0,1)/N
h = 1
k₁₁ = h*κ₂
k₀₁ = (1-h)*κ₂
k₁₀ = (κ₁ - k₁)/2 - k₁₁
results_MF = map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 1/N)[1,end], βs)
plot!(βs, results_MF, width = 4, palette=palette(:Blues)[[9]], label =:none, ylims = [-0.02,0.76])


βs = [0.001:0.001:0.09;]
β△ = 0.2
results_MC_conf_betaTri_02_h_0_ = DelimitedFiles.readdlm("/results/conference/results_MC_conf_betaTri_02_h_0.txt")
results_MC_conf_betaTri_02_h_05_ = DelimitedFiles.readdlm("/results/conference/results_MC_conf_betaTri_02_h_05.txt")
results_MC_conf_betaTri_02_h_1_ = DelimitedFiles.readdlm("/results/conference/results_MC_conf_betaTri_02_h_1.txt")

scatter(βs,results_MC_conf_betaTri_02_h_0_[:,1], width=0., legend =:bottomright,
        label=L"0.0", ms = 7, palette=palette(:Blues)[[3]]);
plot!(βs,results_MC_conf_betaTri_02_h_0_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_conf_betaTri_02_h_0_[:,1] .- results_MC_conf_betaTri_02_h_0_[:,2], results_MC_conf_betaTri_02_h_0_[:,3] .- results_MC_conf_betaTri_02_h_0_[:,1]));
scatter!(βs,results_MC_conf_betaTri_02_h_05_[:,1], width=0.,
        label=L"0.5", ms = 7, palette=palette(:Blues)[[6]]);
plot!(βs,results_MC_conf_betaTri_02_h_05_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_conf_betaTri_02_h_05_[:,1] .- results_MC_conf_betaTri_02_h_05_[:,2], results_MC_conf_betaTri_02_h_05_[:,3] .- results_MC_conf_betaTri_02_h_05_[:,1]));
scatter!(βs,results_MC_conf_betaTri_02_h_1_[:,1], width=0.,
        label=L"1.0", title = L"\beta^{(2)}=%$β△", titlefontsize = 16, xlabel = L"\beta^{(1)}", ylabel = L"I^\star",
        legend_title = L"h", legend_title_font_pointsize = 16, ms = 7, palette=palette(:Blues)[[9]]);
plot!(βs,results_MC_conf_betaTri_02_h_1_[:,1], alpha=0., label=:none, fillalpha=0.5,
        ribbon = (results_MC_conf_betaTri_02_h_1_[:,1] .- results_MC_conf_betaTri_02_h_1_[:,2], results_MC_conf_betaTri_02_h_1_[:,3] .- results_MC_conf_betaTri_02_h_1_[:,1]),
        ylims = [-0.02,0.82])

κ₁ = 2*size(Pairs_h_0,1)/N
κ₂ = 3*size(Triangles_h_0,1)/N
h = 1
k₁₁ = h*κ₂
k₀₁ = (1-h)*κ₂
k₁₀ = (κ₁ - k₁)/2 - k₁₁
results_MF = map(β -> tri_approx(β, β△, k₁, k₁₀, k₀₁, k₁₁, 1/N)[1,end], βs)
plot!(βs, results_MF, width = 4, palette=palette(:Blues)[[9]], label =:none, ylims = [-0.02,0.84])