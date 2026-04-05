using Plots
using LaTeXStrings
using JLD


###
# This script assumes that the script meeting_times_max-reflection_random_walk.jl has already been run
###
N = load("../data/mr_random_walk_meeting_times.jld", "N")
ds = load("../data/mr_random_walk_meeting_times.jld", "ds")
meeting_times = load("../data/mr_random_walk_meeting_times.jld", "meeting_times")
std_err = sqrt.(load("../data/mr_random_walk_meeting_times.jld", "var_meeting_times") ./ N)
h_scale = LinRange(-1.5, -0.5, load("../data/mr_random_walk_meeting_times.jld", "num_hs"))


#Heat map of the meeting times as a function of the time step and dimension
heatmap(ds, -1.5:0.1:-0.5, meeting_times', xscale = :log10, colorbar_scale = :log10)
xlabel!(LaTeXString("Dimension, \$ d\$"))
ylabel!(LaTeXString("Step size, \$ \\log(h)/\\log(d+1)\$"))
plot!(title = "Meeting Time of MR Coupling of Stereo RW")
plot!(colorbar_title = "Expected meeting time")
savefig("rw_expected_meeting_times_heatmap.png")


#Plot of meeting times as a function of dimension
plot();
for (i, h) = enumerate(1.5:-0.1:0.5)
	if i in [1, 3, 5, 8, 11]
		plot!(ds, meeting_times[:, i], ribbon = 2. .* std_err[:, i], marker = :circle, ms = 2, msw = 0.5, label = "h = (d+1)^(-$(h))")
	end
end
plot!(xscale = :log10, yscale = :log10)
plot!(xlabel = LaTeXString("Dimension, \$ d\$"), ylabel = "Expected meeting time", legend_title = "Time step")
plot!(title = "Meeting time of MR coupling of stereo RW")
savefig("rw_expected_meeting_times.png")