using LinearAlgebra
using LoopVectorization
using Distributions
using Random

include("samplers.jl")

function rw_traj(x_0, h, K, RNG)
	"""

	"""
	x_traj = zeros(length(x_0), K)
	x = copy(x_0)
	G = randn(length(x_0))

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		x_traj[:, i] .= stereo_rw_step!(x, h, G)
	end

	return x_traj
end

function parallel_traj(x_0, y_0, h, K, RNG)

	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	x = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		parallel_coupling_step!(x, y, G, G̃, h, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj
end

function reflect_traj(x_0, y_0, h, K, RNG)

	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	x = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		reflection_coupling_step!(x, y, G, G̃, h, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj
end

function mr_traj(x_0, y_0, h, K, RNG)
	
	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj
end