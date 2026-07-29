## Consider the linear regression Y = X β + ϵ
## with ϵ ~ t_ν(0, σ² I), Student's t-distribution with ν degrees of freedom and scale σ² I. We want to estimate β using a Bayesian approach.
## Our prior on β,σ is as follows:
## β ~ N(0, υ I)
## p(σ²) ∝ 1/σ², which is an improper prior on σ²
## The likelihood function is given by 
## p(Y | X, β, σ²) ∝ (σ²)^{-n/2} ∏_{i=1}^n ((1 + (Y_i - X_i' β)²/(ν σ²))^(-(ν+1)/2))

using Random
using Distributions
using Statistics
using Plots
RNG = Xoshiro(1)
include("samplers.jl")
include("adaptive_rwmh.jl")
include("nonadaptive_rwmh.jl")


## simulate data from the model
d = 10
## true beta has zero everywhere except for the first two components
beta_true = [1.0, 2.0]
beta_true = vcat(beta_true, zeros(d - length(beta_true)))
## sample size
n = 100
X = randn(RNG, n, d)
sigma_true = 1.0
nu = 4
ϵ = rand(RNG, TDist(nu), n) .* sigma_true
Y = X * beta_true .+ ϵ
##

## define log-prior density function, upsilon is the prior variance for beta
## Normal prior on beta and improper prior on sigma² > 0 
upsilon = 10
log_prior_density(beta, sigma2) = sigma2 < 0 ? -Inf : -0.5 * dot(beta, beta) / upsilon - log(sigma2)
## alternative: Student prior on beta and same prior on sigma² > 0
# upsilon = 10
# nu_beta = 4
# log_prior_density(beta, sigma2) = sigma2 < 0 ? -Inf : sum(logpdf.(TDist(nu_beta), beta ./ sqrt(upsilon))) .- log(sigma2)


## log-likelihood function associated with linear regression with Student's t-distributed residuals
log_likelihood(beta, sigma2) = sum(logpdf.(TDist(nu), (Y .- X * beta) ./ sqrt(sigma2)) .- 0.5 * log(sigma2))

## define log-posterior density function, theta = [beta; sigma2]
function log_posterior_density(theta)
	beta = theta[1:end-1]
	sigma2 = theta[end]
	log_prior = log_prior_density(beta, sigma2)
	if isinf(log_prior)
		return log_prior # sigma2 < 0, avoid evaluating log_likelihood which requires sigma2 > 0
	end
	return log_likelihood(beta, sigma2) + log_prior
end

function rinit(RNG)
    beta = randn(RNG, d)
    sigma2 = rand(RNG, InverseGaussian(1, 1))
    return [beta; sigma2]
end

theta0 = rinit(RNG)
log_posterior_density(theta0) # check that the log-posterior density is finite at the initial point

## random walk Metropolis-Hastings sampler
## targeting the distribution with density described in log_posterior_density
## the sampler adapts the proposal covariance matrix every 1000 steps to achieve an acceptance rate of 0.234
## (adaptive_rwmh is defined in adaptive_rwmh.jl)

## test adaptive_rwmh

result = adaptive_rwmh(log_posterior_density, theta0, RNG; n_iter = 50000, adapt_every = 2000, log_step_size = -2)
chain_adaptive = result.chain
result.final_acceptance_rate
# list elements of 'result' to see what is returned by adaptive_rwmh
keys(result)
result.final_Sigma



## traceplot of the first component
traceplot = plot(chain_adaptive[:, 1:2], xlabel = "iteration", ylabel = "beta_1", label = false, title = "Traceplot of beta_1")

## side-by-side histograms of the first two components
hist1 = histogram(chain_adaptive[1000:end, 1], xlabel = "beta_1", label = false, title = "Histogram of beta_1")
hist2 = histogram(chain_adaptive[1000:end, 2], xlabel = "beta_2", label = false, title = "Histogram of beta_2")

## traceplot on top, histograms side-by-side below
combined_layout = @layout [a; b c]
combined_plot = plot(traceplot, hist1, hist2, layout = combined_layout)
display(combined_plot)


## run rwmh, starting with final state of chain produced by adaptive_rwmh, and using the final covariance matrix produced by adaptive_rwmh as the proposal covariance matrix
proposal_Sigma = exp(2 * result.log_step_size_seq[end]) * result.final_Sigma

result_rwmh = nonadaptive_rwmh(log_posterior_density, chain_adaptive[end, :], RNG; n_iter = 100_000, proposal_Sigma = proposal_Sigma)
chain_rwmh = result_rwmh.chain
result_rwmh.acceptance_rate

## traceplot of the first component
traceplot = plot(chain_rwmh[:, 1:2], xlabel = "iteration", ylabel = "beta_1", label = false, title = "Traceplot of beta_1")

## side-by-side histograms of the first two components
hist1 = histogram(chain_rwmh[1000:end, 1], xlabel = "beta_1", label = false, title = "Histogram of beta_1")
hist2 = histogram(chain_rwmh[1000:end, 2], xlabel = "beta_2", label = false, title = "Histogram of beta_2")

## traceplot on top, histograms side-by-side below
combined_layout = @layout [a; b c]
combined_plot = plot(traceplot, hist1, hist2, layout = combined_layout)
display(combined_plot)


# ## implement coupled RWMH using "proposal_Sigma"
# proposal_Sigma
# Sigma_chol = cholesky(proposal_Sigma).U
# inv_Sigma_chol = inv(Sigma_chol)



# ## define a function pi0 that takes no arguments and returns a random initial state of the chain
# function pi0()
# 	theta = chain_adaptive[end, :] + randn(RNG, d + 1) .* .1
# 	log_dens_theta = log_posterior_density(theta)
# 	return (theta = theta, logpdf = log_dens_theta)
# end

# function skernel(state)
# 	theta = state.theta
# 	log_dens_theta = state.logpdf
# 	dtheta = length(theta)
# 	theta_prop = theta .+ (Sigma_chol' * randn(RNG, dtheta))
# 	log_dens_prop = log_posterior_density(theta_prop)
# 	if log(rand(RNG)) < log_dens_prop - log_dens_theta
# 		theta = theta_prop
# 		log_dens_theta = log_dens_prop
# 	end
# 	return (theta = theta, logpdf = log_dens_theta)
# end


# state1 = pi0()

# nmcmc = 100_000
# chain_sk = zeros(nmcmc, d + 1)
# for i in 1:nmcmc
# 	state1 = skernel(state1)
# 	chain_sk[i, :] .= state1.theta
# end

# traceplot = plot(chain_sk[1:10_000, 1:5], xlabel = "iteration", ylabel = "beta_1", label = false, title = "Traceplot of beta_1")

# ## compare histogram of chain and histogram of chain_rwmh
# combined_plot = histogram(chain_sk[10_000:end, 2], xlabel = "beta_2", label = "RWMH sk", alpha = 0.5, normalize = true, title = "Histogram of beta_2")
# histogram!(combined_plot, chain_rwmh[10_000:end, 2], label = "RWMH nonadaptive ", alpha = 0.5, normalize = true)
# display(combined_plot)


# include("reflmaxcoupling.jl")
# function ckernel(state1, state2)
# 	identical = false
# 	theta1 = state1.theta
# 	log_dens_theta1 = state1.logpdf
# 	theta2 = state2.theta
# 	log_dens_theta2 = state2.logpdf
# 	reflmax_results = rmvnorm_reflection_max_coupling(theta1, theta2, Sigma_chol, inv_Sigma_chol)
# 	theta_prop1 = reflmax_results.xy[:, 1]
# 	theta_prop2 = reflmax_results.xy[:, 2]
# 	log_dens_prop1 = log_posterior_density(theta_prop1)
# 	log_dens_prop2 = log_posterior_density(theta_prop2)
# 	logu = log(rand(RNG))

# 	accept1 = logu < log_dens_prop1 - log_dens_theta1
# 	accept2 = logu < log_dens_prop2 - log_dens_theta2
# 	if accept1
# 		theta1 = theta_prop1
# 		log_dens_theta1 = log_dens_prop1
# 	end
# 	if accept2
# 		theta2 = theta_prop2
# 		log_dens_theta2 = log_dens_prop2
# 	end
# 	identical = accept1 && accept2 && reflmax_results.identical
# 	state1 = (theta = theta1, logpdf = log_dens_theta1)
# 	state2 = (theta = theta2, logpdf = log_dens_theta2)
# 	return (state1, state2, identical)
# end





# # state2 = pi0()
# # state1, state2, identical = ckernel(state1, state2)

# # state1, state2, identical = ckernel(state1, state2)
# # state1, state2, identical = ckernel(state1, state2)
# # state1, state2, identical = ckernel(state1, state2)
# # state1, state2, identical = ckernel(state1, state2)
# # state1, state2, identical = ckernel(state1, state2)

# include("sample_meeting_time.jl")
# tau = sample_meeting_time(pi0, skernel, ckernel; lag = 1, maxit = 5000)


# ## generate nrep meeting times in parallel 
# nrep = 100
# meeting_times = Vector(undef, nrep)
# for irep in 1:nrep
# 	τ = sample_meeting_time(pi0, skernel, ckernel; lag = 100, maxit = 50_000)
# 	println("irep = $irep, τ = $τ")
# 	meeting_times[irep] = τ
# end

