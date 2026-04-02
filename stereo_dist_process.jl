using LinearAlgebra
using SpecialFunctions
using LoopVectorization
using Distributions
using Random

norm_dist::UnivariateDistribution = Normal()

function transformed_mr_dist_p_inc(y, h, χ_dist, B_dist, RNG)
	U = rand(RNG)
	if U <= 2*ccdf(norm_dist, sqrt(y/(h^2*(2-y))))
		return 0
	else
		X, B = rand(RNG, χ_dist), rand(RNG, B_dist)

		return (1 - h^2*(1 + B)*X/(1 + h^2*X))*y + 2*B*h^2*X/(1 + h^2*X)
	end
end

function transformed_mr_dist_traj!(traj, y_0, h, d, RNG)
	χ_dist = Chi(d/2)
	B_dist = Beta(1/2, d/2)
	y = copy(y_0)
	traj[1] = y
	for i in axes(traj, 1)
		if i >= 2
			y = transformed_mr_dist_p_inc(y, h, χ_dist, B_dist, RNG)
			traj[i] = y
		end
	end

	return traj
end

function transformed_reflect_dist_p_inc(y, h, χ_dist, B_dist, RNG)
	X, B = rand(RNG, χ_dist), rand(RNG, B_dist)

	return (1 - h^2*(1 + B)*X/(1 + h^2*X))*y + 2*B*h^2*X/(1 + h^2*X)
end

function transformed_reflect_dist_traj!(traj, y_0, h, d, RNG)
	χ_dist = Chi(d/2)
	B_dist = Beta(1/2, d/2)
	y = copy(y_0)
	traj[1] = y
	for i in axes(traj, 1)
		if i >= 2
			y = transformed_reflect_dist_p_inc(y, h, χ_dist, B_dist, RNG)
			traj[i] = y
		end
	end

	return traj
end


function simulate_mr_dist_trajs(initial_dist, h, d, time_horizon, N, rng_seed = 0)
	trajs = zeros(time_horizon, N)
	y_0 = 1 - cos(initial_dist)
	Threads.@threads for j in axes(trajs, 2)
		RNG = Xoshiro(rng_seed + j)
		#simulate 1 - cos(dist_k) process. i.e. the transformed dist process
		transformed_mr_dist_traj!(view(trajs, :, j), y_0, h, d, RNG)
	end

	#transform back and return the distance trajectory
	return acos.(1. .- trajs)
end

function simulate_reflect_dist_trajs(initial_dist, h, d, time_horizon, N, rng_seed = 0)
	trajs = zeros(time_horizon, N)
	y_0 = 1 - cos(initial_dist)
	Threads.@threads for j in axes(trajs, 2)
		RNG = Xoshiro(rng_seed + j)
		#simulate 1 - cos(dist_k) process. i.e. the transformed dist process
		transformed_reflect_dist_traj!(view(trajs, :, j), y_0, h, d, RNG)
	end

	#transform back and return the distance trajectory
	return acos.(1. .- trajs)
end
