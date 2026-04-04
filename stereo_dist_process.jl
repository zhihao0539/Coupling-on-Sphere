using LinearAlgebra
using SpecialFunctions
using LoopVectorization
using Distributions
using Random

norm_dist::UnivariateDistribution = Normal()

function mr_dist_inc(y, h, d, RNG)
	U = rand(RNG)
	if U <= 2*ccdf(norm_dist, sqrt(y/(h^2*(2-y))))
		return 0
	else
		υ, ξ = rand(RNG, Chisq(1)), rand(RNG, Chisq(d-1))

		return (y + h^2 * υ*(2 -y))/(1 + h^2 * (υ + ξ))
	end
end

function mr_dist_traj!(traj, y_0, h, d, RNG)
	y = copy(y_0)
	traj[1] = y
	for i in axes(traj, 1)
		if i >= 2
			y = mr_dist_inc(y, h, d, RNG)
			traj[i] = y
		end
	end

	return traj
end

function sim_meeting_times(N, y_0, h, d)
	τs = Array{Float64, 1}(undef, N)

	Threads.@threads for i in axes(τs, 1)
		Y = copy(y_0)
		local RNG = Xoshiro(i)
		local k = 0
		while Y > 0
			Y = mr_dist_inc(Y, h, d, RNG)
			k += 1
		end
		τs[i] = k
	end

	return τs
end


