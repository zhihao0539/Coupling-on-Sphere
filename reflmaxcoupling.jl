using LinearAlgebra
using Random
using Statistics

function rmvnorm_reflection_max_coupling(
    mu1::AbstractVector{<:Real},
    mu2::AbstractVector{<:Real},
    Sigma_chol::AbstractMatrix{<:Real},
    inv_Sigma_chol::AbstractMatrix{<:Real}
)
    d = length(mu1)
    # Scaled difference of means
    scaled_diff = inv_Sigma_chol * (mu2 .- mu1)    
    # Generate d standard normal draws
    xi = randn(d)
    z = -scaled_diff
    normz = norm(z)
    # Store draws in matrix with two columns: first column for x with mean mu1, second column for y with mean mu2
    xy = Matrix{Float64}(undef, d, 2)
    if normz < 1e-15
        identical = true
        xi_out = mu1 .+ Sigma_chol * xi
        xy[:, 1] .= xi_out
        xy[:, 2] .= xi_out
    else
        e = z ./ normz
        utilde = rand()
        edotxi = dot(e, xi)
        # Log-acceptance threshold for reflection maximal coupling
        if log(utilde) < (-0.5 * (edotxi + normz)^2 + 0.5 * edotxi^2)
            eta = xi .+ z
            identical = true
        else
            eta = xi .- 2.0 * edotxi .* e
            identical = false
        end
        # Transform back to target multivariate normal space
        xi_out = mu1 .+ Sigma_chol * xi
        eta_out = mu2 .+ Sigma_chol * eta
        xy[:, 1] .= xi_out
        xy[:, 2] .= identical ? xi_out : eta_out
    end
    return (xy = xy, identical = identical)
end

# ### Test
# ### create a covariance matrix
# Sigma = [3.0 -0.6; -0.6 2.0]
# Sigma_chol = cholesky(Sigma).L
# inv_Sigma_chol = inv(Sigma_chol)
# ## create two mean vectors
# mu1 = [1.2, 2.0]
# mu2 = [-2.0, 0.3]
# ## number of repeats
# n = 1_000_000
# ## sample n times from the reflection maximal coupling of N(mu1, Sigma) and N(mu2, Sigma)
# xy = Matrix{Float64}(undef, 2, 2 * n)
# identical = Vector{Bool}(undef, n)
# for i in 1:n
#     result = rmvnorm_reflection_max_coupling(mu1, mu2, Sigma_chol, inv_Sigma_chol)
#     xy[:, 2 * i - 1:2 * i] = result.xy
#     identical[i] = result.identical
# end
# ## obtain x and y samples
# x_samples = xy[:, 1:2:end]
# y_samples = xy[:, 2:2:end]
# ## mean of x_samples and y_samples
# mean_x = vec(mean(x_samples, dims = 2))
# mean_y = vec(mean(y_samples, dims = 2))
# ## covariance of x_samples and y_samples
# cov_x = cov(x_samples, dims = 2)
# cov_y = cov(y_samples, dims = 2)
# ## proportion of identical samples
# prop_identical = mean(identical)
