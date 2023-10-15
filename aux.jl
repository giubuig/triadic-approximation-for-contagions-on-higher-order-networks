using Distributed
# addprocs(6)
@everywhere using StatsBase, ProgressMeter, Random

@everywhere function montecarlo_qs(N, Pairs, Triangles, β, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p)
    inf = zeros(Int16,N)
    s = sample([1:N;], Int(p₀*N))
    inf[s] .= 1
  
    L = size(Pairs, 1)
    T = size(Triangles, 1)
  
    stored_states = zeros(Int64, N, M)
    for m in 1:M
      stored_states[:,m] = shuffle(inf)
    end
  
    temp_inf = copy(inf)
  
    ρ_evo = zeros(Float64, max_timesteps)
    ρ_evo[1] = mean(inf)
  
    ω = ones(N)
  
    @inbounds @showprogress 1 "time" for t in 2:max_timesteps
      fill!(ω,1)
      @simd for indx in 1:L
        i, j = Pairs[indx]
        ω[i] *= 1 - Δt*β*inf[j]
        ω[j] *= 1 - Δt*β*inf[i]
      end
      @simd for indx in 1:T
        i, j, l = Triangles[indx]
        ω[i] *= 1 - Δt*β△*inf[j]*inf[l]
        ω[j] *= 1 - Δt*β△*inf[i]*inf[l]
        ω[l] *= 1 - Δt*β△*inf[i]*inf[j]
      end
  
      @simd for i in 1:N
        r = rand()
        if inf[i] == 0
          if r < 1 - ω[i]
            temp_inf[i] = 1
          end
        else
          if r < μ*Δt
            temp_inf[i] = 0
          end
        end
      end
  
      ρ_evo[t] = mean(temp_inf)
  
      if sum(temp_inf) == 0
        inf .= stored_states[:,rand(1:M)]
      else
        inf .= temp_inf
        if rand() < update_p
          rand_q = rand(1:M)
          for i in 1:N
            stored_states[i, rand_q] = inf[i]
          end
        end
      end
    end
  
    avg_rho = mean(ρ_evo[(max_timesteps - average_window):end])
    println(avg_rho)
    return avg_rho, ρ_evo
end

@everywhere function montecarlo_qs_SIR(N, Pairs, Triangles, β, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p)
  σ = zeros(Int16,N)
  s = sample([1:N;], Int(p₀*N))
  σ[s] .= 1

  L = size(Pairs, 1)
  T = size(Triangles, 1)

  stored_states = zeros(Int64, N, M)
  for m in 1:M
    stored_states[:,m] = shuffle(σ)
  end

  temp_σ = copy(σ)

  ρ_evo = zeros(Float64, max_timesteps, 2)
  ρ_evo[1,1] = count(x -> x == 1, σ)/N
  ρ_evo[1,2] = 0.

  ω = ones(N)

  @inbounds @showprogress 1 "time" for t in 2:max_timesteps
    fill!(ω,1)
    @simd for indx in 1:L
      i, j = Pairs[indx]
      ω[i] *= 1 - Δt*β*(σ[j] == 1)
      ω[j] *= 1 - Δt*β*(σ[i] == 1)
    end
    @simd for indx in 1:T
      i, j, l = Triangles[indx]
      ω[i] *= 1 - Δt*β△*(σ[j] == 1 && σ[l] == 1)
      ω[j] *= 1 - Δt*β△*(σ[i] == 1 && σ[l] == 1)
      ω[l] *= 1 - Δt*β△*(σ[i] == 1 && σ[j] == 1)
    end

    @simd for i in 1:N
      r = rand()
      if σ[i] == 0
        if r < 1 - ω[i]
          temp_σ[i] = 1
        end
      elseif σ[i] == 1
        if r < μ*Δt
          temp_σ[i] = 2
        end
      end
    end

    ρ_evo[t,1] = count(x -> x == 1, temp_σ)/N
    ρ_evo[t,2] = count(x -> x == 2, temp_σ)/N

    if ρ_evo[t,1] == 0. && ρ_evo[t,2] < p₀ + 1/N
      σ .= stored_states[:,rand(1:M)]
    else
      σ .= temp_σ
      if rand() < update_p
        rand_q = rand(1:M)
        for i in 1:N
          stored_states[i, rand_q] = σ[i]
        end
      end
    end
  end

  final_attack = ρ_evo[max_timesteps,2]
  println(final_attack)
  return final_attack, ρ_evo
end

@everywhere function regular_3_complex(N, k₁, k₁₀, k₀₁, k₁₁)
  link_vec = repeat([1:N;],k₁)
  clique3_vec = repeat([1:N;],k₁₀)
  edge3_vec = repeat([1:N;],k₀₁)
  simp2_vec = repeat([1:N;],k₁₁)
  Pairs = Vector{Array{Int64,1}}()
  Triangles = Vector{Array{Int64,1}}()

  while length(simp2_vec) > 3
    samp = sort!(sample(simp2_vec,3,replace=false))
    push!(Triangles, samp)
    push!(Pairs, samp[[1;2]])
    push!(Pairs, samp[[1;3]])
    push!(Pairs, samp[[2;3]])
    for i in 1:3
        deleteat!(simp2_vec, findfirst(x -> x == samp[i], simp2_vec))
    end
  end
  if length(simp2_vec) > 2
    push!(Triangles, simp2_vec)
    push!(Pairs, simp2_vec[[1;2]])
    push!(Pairs, simp2_vec[[1;3]])
    push!(Pairs, simp2_vec[[2;3]])
  end

  while length(edge3_vec) > 3
    samp = sort!(sample(edge3_vec,3,replace=false))
    push!(Triangles, samp)
    for i in 1:3
        deleteat!(edge3_vec, findfirst(x -> x == samp[i], edge3_vec))
    end
  end
  if length(edge3_vec) > 2
    push!(Triangles, edge3_vec)
  end

  while length(clique3_vec) > 3
    samp = sort!(sample(clique3_vec,3,replace=false))
    push!(Pairs, samp[[1;2]])
    push!(Pairs, samp[[1;3]])
    push!(Pairs, samp[[2;3]])
    for i in 1:3
        deleteat!(clique3_vec, findfirst(x -> x == samp[i], clique3_vec))
    end
  end
  if length(clique3_vec) > 2
    push!(Pairs, clique3_vec[[1;2]])
    push!(Pairs, clique3_vec[[1;3]])
    push!(Pairs, clique3_vec[[2;3]])
  end

  while length(link_vec) > 2
    samp = sort!(sample(link_vec,2,replace=false))
    push!(Pairs, samp)
    for i in 1:2
          deleteat!(link_vec, findfirst(x -> x == samp[i], link_vec))
    end
  end
  if length(link_vec) > 1
    push!(Pairs, link_vec)
  end

  return Pairs, Triangles
end

@everywhere function get_results_MC(N, k₁, k₁₀, k₀₁, k₁₁, βs, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p, n)
  res = zeros(Float64,length(βs),2)
  Pairs, Triangles = regular_3_complex(N, k₁, k₁₀, k₀₁, k₁₁)
  for indx in 1:length(βs)
      β = βs[indx]
      println(β)
      r = pmap(x -> montecarlo_qs(N, Pairs, Triangles, x, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p)[1], repeat([β],n))
      res[indx,1] = mean(r)
      res[indx,2] = std(r)
  end
  return res
end

@everywhere function get_results_MC_SIR(N, k₁, k₁₀, k₀₁, k₁₁, βs, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p, n)
  res = zeros(Float64,length(βs),2)
  res_det = zeros(Float64,length(βs),n)
  Pairs, Triangles = regular_3_complex(N, k₁, k₁₀, k₀₁, k₁₁)
  for indx in 1:length(βs)
      β = βs[indx]
      println(β)
      r = pmap(x -> montecarlo_qs_SIR(N, Pairs, Triangles, x, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p)[1], repeat([β],n))
      res[indx,1] = mean(r)
      res[indx,2] = std(r)
      res_det[indx, :] = r
  end
  return res, res_det
end

@everywhere function get_results_MC_real_net(N, Pairs, Triangles, βs, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p, n)
  res = zeros(Float64,length(βs),2)
  res_det = zeros(Float64,length(βs),n)
  for indx in 1:length(βs)
      β = βs[indx]
      println(β)
      r = pmap(x -> montecarlo_qs(N, Pairs, Triangles, x, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p)[1], repeat([β],n))
      res[indx,1] = mean(r)
      res[indx,2] = std(r)
      res_det[indx, :] = r
  end
  return res, res_det
end

@everywhere function get_results_MC_real_net_SIR(N, Pairs, Triangles, βs, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p, n)
  res = zeros(Float64,length(βs),2)
  res_det = zeros(Float64,length(βs),n)
  for indx in 1:length(βs)
      β = βs[indx]
      println(β)
      r = pmap(x -> montecarlo_qs_SIR(N, Pairs, Triangles, x, β△, μ, p₀, max_timesteps, Δt, average_window, M, update_p)[1], repeat([β],n))
      res[indx,1] = mean(r)
      res[indx,2] = std(r)
      res_det[indx, :] = r
  end
  return res, res_det
end