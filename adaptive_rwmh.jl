using LinearAlgebra
using Statistics

## random walk Metropolis-Hastings sampler targeting the distribution with density log_density
## the sampler adapts the proposal covariance matrix every adapt_every steps to achieve an acceptance rate of target_acceptance
function adaptive_rwmh(log_density, theta0, RNG; n_iter, log_step_size = 0.0, adapt_every = 1000, target_acceptance = 0.234, Sigma = Matrix{Float64}(I, length(theta0), length(theta0)))
	d_theta = length(theta0)
	chain = zeros(n_iter, d_theta)
	theta = copy(theta0)
	log_dens_theta = log_density(theta)

	cholSigma = cholesky(Sigma).L
	n_accepts_batch = 0
	n_batches = n_iter ÷ adapt_every
	log_step_size_seq = zeros(n_batches)
	final_acceptance_rate = 0.0

	for iter in 1:n_iter
		theta_prop = theta .+ exp(log_step_size) .* (cholSigma * randn(RNG, d_theta))
		log_dens_prop = log_density(theta_prop)

		if log(rand(RNG)) < log_dens_prop - log_dens_theta
			theta = theta_prop
			log_dens_theta = log_dens_prop
			n_accepts_batch += 1
		end
		chain[iter, :] .= theta

		if iter % adapt_every == 0
			batch = @view chain[(iter - adapt_every + 1):iter, :]
			Sigma = Symmetric(cov(batch)) + 1e-8 * I # regularize to keep the covariance estimate positive definite
			cholSigma = cholesky(Sigma).L
			final_acceptance_rate = n_accepts_batch / adapt_every
			# diminishing adaptation of the log step size, Robbins-Monro style, targeting the acceptance rate
			log_step_size += (final_acceptance_rate - target_acceptance) / sqrt(iter / adapt_every)
			log_step_size_seq[iter ÷ adapt_every] = log_step_size
			n_accepts_batch = 0
		end
	end

	return (chain = chain, log_step_size_seq = log_step_size_seq, final_acceptance_rate = final_acceptance_rate, final_Sigma = Sigma)
end
