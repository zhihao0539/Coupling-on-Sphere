using LinearAlgebra
using SpecialFunctions
using LoopVectorization
using Distributions
using Random
using ProgressBars

include("samplers.jl")


function sim_meeting_times(N, h, d, R, log_density; max_iter = 1e6)
	τs = Array{Float64, 1}(undef, N)
	x_acceptance_proba = Array{Float64, 1}(undef, N)
	y_acceptance_proba = Array{Float64, 1}(undef, N)

	Threads.@threads for i in tqdm(axes(τs, 1))
		RNG = Xoshiro(i)
		x = normalize(randn(RNG, d+1))
		x_prop = copy(x)
		X_new = copy(x)
		y = -copy(x)
		y_prop = copy(y)
		G = randn(RNG, d+1)
		G̃ = randn(RNG, d+1)
		k = 0

		x_acceptance_count = 0.
		y_acceptance_count = 0.
		while x != y && k <= max_iter
			x_accepted, y_accepted = coupled_MH_step!(x, x_prop, y, y_prop, X_new, G, G̃, h, d, R, log_density, RNG)
			k += 1
			x_acceptance_count += x_accepted
			y_acceptance_count += y_accepted
		end
		τs[i] = k
		x_acceptance_proba[i] = x_acceptance_count/k
		y_acceptance_proba[i] = y_acceptance_count/k
	end

	return τs, x_acceptance_proba, y_acceptance_proba
end