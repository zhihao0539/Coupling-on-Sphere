using LinearAlgebra
using Distributions
using Random
using ProgressBars
using Plots


ds = [5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 7500, 10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000, 200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000]
h_scale = 0.01:0.01:1.0
ps = 1:1:30
N = 10_000_000

factor(υ, ξ; h) = (1 + h^2 * υ)/(1 + h^2 *(υ + ξ))
factor_moments = zeros(length(h_scale), length(ds), length(ps))

for (j, d) in tqdm(enumerate(ds))
	U, X = rand(Chisq(1), N), rand(Chisq(d-1), N)
	Threads.@threads for i in axes(h_scale, 1)
		factor_moments[i, j, :] .= mean(factor.(U, X; h = d^(-1/2) * h_scale[i]) .^ (ps ./ 2)', dims =1)' .^ (1 ./ ps)
	end
end

coef = -1. ./ log.(factor_moments)

scaled_coef = zero(coef)
for (i, h) in enumerate(h_scale), (j, d) in enumerate(ds), (k, p) in enumerate(ps)
	scaled_coef[i, j, k] = h^2 * coef[i,j,k]
end