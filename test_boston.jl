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
using CSV
using DataFrames
RNG = Xoshiro(1)
include("samplers.jl")
include("adaptive_rwmh.jl")
include("nonadaptive_rwmh.jl")


## import BostonHousing.csv file
data = CSV.read("BostonHousing.csv", DataFrame)

Y = Float64.(data.medv)
X = Matrix{Float64}(select(data, Not(:medv)))
X = (X .- mean(X, dims = 1)) ./ std(X, dims = 1) # standardize each column to mean 0, std 1
X = hcat(ones(size(X, 1)), X) # add column of ones for the intercept

n, d = size(X)


## define log-prior density function, upsilon is the prior variance for beta
upsilon = 10
## Normal prior on beta and improper prior on sigma² > 0 
log_prior_density(beta, sigma2) = sigma2 < 0 ? -Inf : -0.5 * dot(beta, beta) / upsilon - log(sigma2)
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
    beta = 10 .* randn(RNG, d)
    sigma2 = rand(RNG, InverseGaussian(1, 1))
    return [beta; sigma2]
end

theta0 = rinit(RNG)
log_posterior_density(theta0) # check that the log-posterior density is finite at the initial 

## test adaptive RWMH
result = adaptive_rwmh(log_posterior_density, theta0, RNG; n_iter = 50000, adapt_every = 2000, log_step_size = -2)
chain = result.chain
println(result.final_acceptance_rate)

## traceplot of the first component
traceplot = plot(chain[:, 1:2], xlabel = "iteration", ylabel = "beta_1", label = false, title = "Traceplot of beta_1")

## side-by-side histograms of the first two components
hist1 = histogram(chain[1000:end, 1], xlabel = "beta_1", label = false, title = "Histogram of beta_1")
hist2 = histogram(chain[1000:end, 2], xlabel = "beta_2", label = false, title = "Histogram of beta_2")

## traceplot on top, histograms side-by-side below
combined_layout = @layout [a; b c]
combined_plot = plot(traceplot, hist1, hist2, layout = combined_layout)
display(combined_plot)


proposal_Sigma = exp(2 * result.log_step_size_seq[end]) * result.final_Sigma

result_rwmh = nonadaptive_rwmh(log_posterior_density, chain[end, :], RNG; n_iter = 10000, proposal_Sigma = proposal_Sigma)
chain_rwmh = result_rwmh.chain
result_rwmh.acceptance_rate

## traceplot of the first component
traceplot = plot(chain_rwmh[:, 1:4], xlabel = "iteration", ylabel = "beta_1", label = false, title = "Traceplot of beta_1")

## side-by-side histograms of the first two components
hist1 = histogram(chain_rwmh[1000:end, 1], xlabel = "beta_1", label = false, title = "Histogram of beta_1")
hist2 = histogram(chain_rwmh[1000:end, 2], xlabel = "beta_2", label = false, title = "Histogram of beta_2")

## side-by-side histograms of the first two components
hist3 = histogram(chain_rwmh[1000:end, 3], xlabel = "beta_3", label = false, title = "Histogram of beta_3")
hist4 = histogram(chain_rwmh[1000:end, 4], xlabel = "beta_4", label = false, title = "Histogram of beta_4")


## traceplot on top, histograms side-by-side below
combined_layout = @layout [a; b c; d e]
combined_plot = plot(traceplot, hist1, hist2, hist3, hist4, layout = combined_layout)
display(combined_plot)
