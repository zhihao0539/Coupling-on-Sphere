using LinearAlgebra
using Statistics
## random walk Metropolis-Hastings sampler targeting the distribution with density log_density
## the sampler takes a proposal covariance matrix as input

function nonadaptive_rwmh(log_density, theta0, RNG; n_iter, proposal_Sigma = Matrix{Float64}(I, length(theta0), length(theta0)))
	d_theta = length(theta0)
	chain = zeros(n_iter, d_theta)
	theta = copy(theta0)
	log_dens_theta = log_density(theta)

	cholSigma = cholesky(proposal_Sigma).L
	n_accepts = 0
	acceptance_rate = 0.0

	for iter in 1:n_iter
		theta_prop = theta .+ (cholSigma * randn(RNG, d_theta))
		log_dens_prop = log_density(theta_prop)

		if log(rand(RNG)) < log_dens_prop - log_dens_theta
			theta = theta_prop
			log_dens_theta = log_dens_prop
			n_accepts += 1
		end
		chain[iter, :] .= theta
	end
	acceptance_rate = n_accepts / n_iter

	return (chain = chain, acceptance_rate = acceptance_rate)
end
