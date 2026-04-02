using LinearAlgebra
using Distributions
using Random
using ProgressBars
using Plots
using LoopVectorization

function sim_stationary_traj(h, d, K)

	#generate noise for trajectory
	υ, ξ = rand(Chisq(1), K), rand(Chisq(d-1), K)
	traj = Array{Float64, 1}(undef, K)

	A = (1. .- h^2 .* υ) ./ (1. .+ h^2 .* (υ .+ ξ))

	for k in axes(traj, 1)
		traj[k] = (2. * h^2) * sum(υ[i]/(1 + h^2 * (υ[i] + ξ[i])) * prod(view(A, (i+1):k)) for i = 1:k)
	end

	return traj
end

function sim_stationary_traj_treads(h, d, K)

	#generate noise for trajectory
	υ, ξ = rand(Chisq(1), K), rand(Chisq(d-1), K)
	traj = Array{Float64, 1}(undef, K)

	A = (1. .- h^2 .* υ) ./ (1. .+ h^2 .* (υ .+ ξ))

	Threads.@threads for k in axes(traj, 1)
		traj[k] = (2. * h^2) * sum(υ[i]/(1 + h^2 * (υ[i] + ξ[i])) * prod(view(A, (i+1):k)) for i = 1:k)
	end

	return traj
end