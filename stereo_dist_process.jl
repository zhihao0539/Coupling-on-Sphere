using LinearAlgebra
using SpecialFunctions
using LoopVectorization
using Distributions
using Random

include("samplers.jl")



function reflect_traj(x_0, y_0, h, K, RNG)

	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	υ_traj = zeros(K)
	ξ_traj = zeros(K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		υ_traj[i] = dot(get_e_x_perp_y(x,y), G)^2
		ξ_traj[i] = norm(G .- dot(x, G) .* x .- dot(get_e_x_perp_y(x,y), G) .* get_e_x_perp_y(x,y))^2

		reflection_coupling_step!(x, y, G, G̃, h, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj, υ_traj, ξ_traj
end

function parallel_traj_test(x_0, y_0, h, K, RNG)

	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	υ_traj = zeros(K)
	ξ_traj = zeros(K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		υ_traj[i] = dot(get_e_x_perp_y(x,y), G)^2
		ξ_traj[i] = norm(G .- dot(x, G) .* x .- dot(get_e_x_perp_y(x,y), G) .* get_e_x_perp_y(x,y))^2

		parallel_coupling_step!(x, y, G, G̃, h, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj, υ_traj, ξ_traj
end

function reflect_dist_traj(x_0, y_0, h, υ, ξ)


	dist_traj = zeros(length(υ))
	Y = 1. .- dot(x_0, y_0)

	for i in axes(υ, 1)
		Y = ((1. - h^2 * υ[i])*Y + 2. * h^2)/(1 + h^2 * (υ[i] + ξ[i]))
		dist_traj[i] = Y
	end

	return dist_traj
end

function mr_traj_test(x_0, y_0, h, K, RNG)
	
	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	υ_traj = zeros(K)
	ξ_traj = zeros(K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		υ_traj[i] = dot((y .- dot(x,y).*x), G)/sqrt(1 - dot(x,y)^2)
		ξ_traj[i] = norm(G .- dot(G, x) .* x .- (dot((y .- dot(x,y).*x), G)/(1 - dot(x,y)^2)) .* (y .- dot(x,y).*x))^2

		max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj, υ_traj, ξ_traj
end

function sim_meeting_times(N, h, d)
	τs = Array{Float64, 1}(undef, N)

	Threads.@threads for i in tqdm(axes(τs, 1))
		RNG = Xoshiro(i)
		x = normalize(randn(RNG, d+1))
		X_new = copy(x)
		y = -copy(x)
		G = randn(RNG, d+1)
		G̃ = randn(RNG, d+1)
		k = 0
		while x != y
			max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
			k += 1
		end
		τs[i] = k
	end

	return τs
end


