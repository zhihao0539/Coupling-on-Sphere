using Random
using LinearAlgebra

#negative log of the meeting probability
g(x::Vector{Float64}, y::Vector{Float64}; h::Float64) = -logccdf(Normal(), sqrt((1 - dot(x,y))/(1 + dot(x,y)))/h) - log(2)


function P(x, y, v)
	"""
	Parallel transport operator on the sphere
	"""
	if x != -y
		return v .- (x .+ y) .* (dot(y, v)/(1 + dot(x,y)))
	else
		#special case for when x and y are antipodal
		local pₛ = zeros(length(x))
		pₛ[end] = -1.
		return v .- 2. .* (pₛ .- dot(pₛ,x) .* x) .* (dot(pₛ, v)/(1 - dot(x, pₛ)^2))
	end
end

function M(x, y, v)
	"""
	Parallel transport and reflection operator on the sphere
	"""
	if x == y
		return v
	elseif x != -y
		return v .- (x .+ y) .* (dot(y, v)/(1 + dot(x,y))) .+ (2*dot(y .- (x⋅y).*x, v)/(1 - dot(x,y)^2)) .* (x .- (x⋅y).*y)
	else
		#special case for when x and y are antipodal
		local pₛ = zeros(length(x))
		pₛ[end] = -1.
		return v .- 2. .* (pₛ .- dot(pₛ,x) .* x) .* (dot(pₛ, v)/(1 - dot(x, pₛ)^2)) .+ (2*dot(pₛ .- (x⋅pₛ).*x, v)/(1 - dot(x,pₛ)^2)) .* (pₛ .- (y⋅pₛ).*y)
	end

end

function get_e_x_perp_y(x, y)
	"""
	Compute the unit vector in the tangent space of x pointing to y
	"""

	if dot(x,y) >= 1.
		return x .- y
	elseif dot(x,y) > -1
		return (y .- dot(x,y) .* x)./sqrt(1 - dot(x,y)^2)
	else
		#special case for when x and y are antipodal in which case all unit vectors in the tangent 
		#space of x point to y so we choose the one pointing to the south pole
		local pₛ = zeros(length(x))
		pₛ[end] = -1.
		return (pₛ .- dot(x,pₛ) .* x)./sqrt(1 - dot(x,pₛ)^2)
	end
end

function stereo_rw_step!(x, h, G)
	"""

	"""
	x .+= h .* (G .- dot(G, x).* x)
	normalize!(x)
	return x
end


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

function parallel_coupling_step!(x, y, G, G̃, h, RNG)
	"""

	"""
	G .-= dot(G, x) .* x #projecting noise onto tangent space of x
	G̃ .= P(x, y, G) #parallel transporting noise to tangent space of y
	x .+= h .* G
	y .+= h .* G̃
	normalize!(x)
	normalize!(y)

	return x, y
end

function reflection_coupling_step!(x, y, G, G̃, h, RNG)
	"""

	"""
	G .-= dot(G, x) .* x #projecting noise onto tangent space of x
	G̃ .= M(x, y, G) #parallel transporting noise to tangent space of y and reflecting it
	x .+= h .* G
	y .+= h .* G̃
	normalize!(x)
	normalize!(y)

	return x, y
end

function max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
	"""

	"""
	G .-= dot(G, x) .* x #projecting noise onto tangent space of x
	G̃ .= M(x, y, G) #parallel transporting noise to tangent space of y and reflecting it

	X_new .+= h .* G
	normalize!(X_new)

	if (dot(y, X_new) >= 0.) && (randexp(RNG) >= (d+1) * log(dot(x, X_new)/dot(y, X_new)) + (1/dot(x, X_new)^2 - 1/dot(y, X_new)^2)/(2. *h^2))
		#meeting event
		y .= X_new
	else
		#reflection event
		y .+= h .* G̃
		normalize!(y)
	end
	x .= X_new

	return x, y
end


function reflect_traj(x_0, y_0, h, K, RNG)

	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	υ_traj = zeros(K)
	ξ_traj = zeros(K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		υ_traj[i] = dot(get_e_x_perp_y(x,y), G)^2
		ξ_traj[i] = norm(G .- dot(x, G) .* x .- dot(get_e_x_perp_y(x,y), G) .* get_e_x_perp_y(x,y))^2

		reflection_coupling_step!(x, y, G, G̃, h, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj, υ_traj, ξ_traj
end

function parallel_traj_test(x_0, y_0, h, K, RNG)

	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	υ_traj = zeros(K)
	ξ_traj = zeros(K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		υ_traj[i] = dot(get_e_x_perp_y(x,y), G)^2
		ξ_traj[i] = norm(G .- dot(x, G) .* x .- dot(get_e_x_perp_y(x,y), G) .* get_e_x_perp_y(x,y))^2

		parallel_coupling_step!(x, y, G, G̃, h, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj, υ_traj, ξ_traj
end

function reflect_dist_traj(x_0, y_0, h, υ, ξ)


	dist_traj = zeros(length(υ))
	Y = 1. .- dot(x_0, y_0)

	for i in axes(υ, 1)
		Y = ((1. - h^2 * υ[i])*Y + 2. * h^2)/(1 + h^2 * (υ[i] + ξ[i]))
		dist_traj[i] = Y
	end

	return dist_traj
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

function mr_traj_test(x_0, y_0, h, K, RNG)
	
	x_traj = zeros(length(x_0), K)
	y_traj = zeros(length(y_0), K)

	υ_traj = zeros(K)
	ξ_traj = zeros(K)

	x = copy(x_0)
	X_new = copy(x_0)
	y = copy(y_0)
	G = randn(length(x_0))
	G̃ = randn(length(x_0))

	d = length(x_0) - 1 #dimension of the sphere

	for i in axes(x_traj, 2)
		randn!(RNG, G)
		υ_traj[i] = dot((y .- dot(x,y).*x), G)/sqrt(1 - dot(x,y)^2)
		ξ_traj[i] = norm(G .- dot(G, x) .* x .- (dot((y .- dot(x,y).*x), G)/(1 - dot(x,y)^2)) .* (y .- dot(x,y).*x))^2

		max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
		x_traj[:, i] .= x
		y_traj[:, i] .= y
	end

	return x_traj, y_traj, υ_traj, ξ_traj
end

function sim_meeting_times(N, h, d)
	τs = Array{Float64, 1}(undef, N)

	Threads.@threads for i in axes(τs, 1)
		RNG = Xoshiro(i)
		x = normalize(randn(RNG, d))
		X_new = copy(x)
		y = .-copy(x)
		G = randn(RNG, d)
		G̃ = randn(RNG, d)
		k = 0
		while x != y
			max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
			k += 1
		end
		τs[i] = k
	end

	return τs
end