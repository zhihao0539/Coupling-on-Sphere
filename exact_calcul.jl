using LinearAlgebra
using SpecialFunctions
using LoopVectorization

@inline function integrand_func(u, s, h, d)
	return (1 + h^2 * u)^(d/2) * exp(-u/2) /sqrt(u*(u - s))
end

function I(s; h, d, L, dv, du)
	integral = 0.
	vs = (s:dv:(1. + s))[2:end] #mesh for fine scale, i.e. near s
	@turbo for i in axes(vs, 1)
		integral += integrand_func(vs[i], s, h, d)*dv
	end

	us = vs[end]:du:(L + s) #mesh for coarse scale
	@turbo for i in axes(us, 1)
		integral += integrand_func(us[i], s, h, d)*du
	end
	

	return sqrt(1 + h^2*s)*integral
end


function ψ(u; h, d, L = 1e2, dv = 1e-6, du = 1e-3)
	"""
	Density of increments
	"""
	return exp(-u)*(1 - exp(-u))^(d/2 - 1)*I((exp(u) - 1)/h; h = h, d = d, L=L, dv = dv, du = du)/(2^(d/2)*h^d * gamma(d/2)*beta(d/2, 1/2))
end

function exp_fd(u, h; tol = 1e-14)
	"""
	Approximation of 

	\frac{e^{u h^2}-1}{h^2} = u + u^2h^2/2 + u^3h^4/6 + ...
	"""

	if u^2*h^3 <= tol 
		return u + u^2*h^2/2 + u^3*h^4/6
	else
		return expm1(u*h^2)/h^2
	end
end

function exp_bd(u, h; tol =1e-14)
	"""
	Approximation of 

	\frac{1 - e^{-u h^2}}{h^2} = u - u^2h^2/2 + u^3h^4/6 - ...
	"""
	if u^2*h^3 <= tol 
		return u - u^2*h^2/2 + u^3*h^4/6
	else
		return -expm1(-u*h^2)/h^2
	end
end

function ψ_h(u; h, d, L = 1e2, dv = 1e-6, du = 1e-3)
	"""
	Density of increments rescaled by h, i.e.
	ψ_h(u) = h^2 ψ(h^2u)
	"""
	return exp(-h^2*u)*(exp_bd(u, h))^(d/2 - 1)*I(exp_fd(u,h); h = h, d = d, L=L, dv = dv, du = du)/(2^(d/2) * gamma(d/2)*beta(d/2, 1/2))
end

function facteur_density(u; h, d, L = 1e2, dv = 1e-6, du = 1e-3)
	return (1 -u)^(d/2 - 1)*I((1-u)/(h^2*u); h = h, d = d, L=L, dv = dv, du = du)/(2^(d/2)*h^d * gamma(d/2)*beta(d/2, 1/2))
end