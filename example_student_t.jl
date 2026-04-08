using JLD
using Plots
using LaTeXStrings

include("coupled_stereo_mrw_meeting_times.jl")


N = 10_000
h::Float64, d::Int64 = 0.01, 100
ℓs = 0.125:0.125:6.
Rs = 1.:20.

ν::Int64 = d + 10
log_density(x) = ((ν + d)/2) * log(1 + dot(x,x)/ν)
τs = zeros(N, length(ℓs), length(Rs))
x_accept_rate = zeros(N, length(ℓs), length(Rs))
y_accept_rate = zeros(N, length(ℓs), length(Rs))

for (i, ℓ) in enumerate(ℓs), (j, R) in enumerate(Rs)
	τ, x_acc, y_acc = sim_meeting_times(N, ℓ*h, d, R, log_density)
	τs[:, i, j] .= τ
	x_accept_rate[:, i, j] .= x_acc
	y_accept_rate[:, i, j] .= y_acc
end

save("example_student_d$(d)_nu$(ν).jld", "N", N, "h", h, "d", d, "nu", ν, "Rs", Rs, "ls", ℓs, "meeting_times", τs, "x_accept_rate", x_accept_rate, "y_accept_rate", y_accept_rate)

heatmap(Rs, ℓs, [mean(τs[:, i, j]) for i in axes(τs, 2), j in axes(τs, 3)], colorbar_scale = :log10, xlabel = "Stereographic Projection Parameter, \$ R\$", ylabel = "Step size, \$\\ell/d\$", title = "Average meeting time, \$\\log\\left(\\pi(x)\\right) = -\\log(1 + |x|^2)\$")
savefig("figures/example_student_d$(d)_nu$(ν)_avg_meeting_time.png")

heatmap(Rs, ℓs, [quantile(τs[:, i, j], 0.75) for i in axes(τs, 2), j in axes(τs, 3)], xlabel = "Stereographic Projection Parameter, \$ R\$", ylabel = "Step size, \$\\ell/d\$", title = "75% of meeting time, \$\\log\\left(\\pi(x)\\right) = -\\log(1 + |x|^2)\$")
savefig("figures/example_student_d$(d)_nu$(ν)_75percentile_meeting_time.png")