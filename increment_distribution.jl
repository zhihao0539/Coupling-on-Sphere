using LaTeXStrings
using ProgressBars
include("exact_calcul.jl")

us = 0.0:0.001:4.
n_subinterval = 4
ds = [10, 25, 50, 100, 200, 300]
hs = [0.01, 0.02, 0.05, 0.1, 0.2]
pdf = zeros(length(us), length(ds), length(hs))
moments = zeros(5, length(ds), length(hs))
exp_moments = zeros(4, length(ds), length(hs))

for (k, d) in enumerate(ds)
       println("d = $(d)")
       for (l, h) in tqdm(enumerate(hs)) 
              Threads.@threads for i in 1:1000
                     pdf[i, k, l] =  d .* ψ_h(us[i] * d; h = h, d=d)
              end
              if maximum(pdf[990:1000, k, l]) > 0
                     Threads.@threads for i in 1001:4001
                            pdf[i, k, l] =  d .* ψ_h(us[i] * d; h = h, d=d)
                     end
              end
       end
end

pdf[isnan.(pdf)] .= 0.

for (j, d) in enumerate(ds)
       for (k, h) in enumerate(hs)
              moments[1, j, k] =  us[2] * (sum(pdf[2:end-1, j, k]) + (pdf[1, j, k] + pdf[end, j, k])/2)
       end
end

for i in axes(exp_moments, 1)
       for (j, d) in enumerate(ds)
              for (k, h) in enumerate(hs)
                     moments[i+1, j, k] = (h^2 * d)^i * us[2] * (sum(us[2:end-1].^i .* pdf[2:end-1, j, k]) + (us[1]^i * pdf[1, j, k] + us[end]^i * pdf[end, j, k])/2)
                     exp_moments[i, j, k] = us[2] * (sum(exp.(-(i * h^2 * d) .* us[2:end-1]) .* pdf[2:end-1, j, k]) + (exp.(-(i * h^2 * d) * us[1]) * pdf[1, j, k] + exp.(-(i * h^2 * d) * us[end]) * pdf[end, j, k])/2)
              end
       end
end

plot() 
for (i, d) in enumerate(ds)
       j, h = 4, 0.1
       plot!(exp.(-h^2*d*us[2:end]), pdf[2:end, i, j], label = "h = $(h), d = $(d)", color = palette(:default)[i])
       j, h = 5, 0.2
       plot!(exp.(-h^2*d*us[2:end]), pdf[2:end, i, j], ls = :dash, label = "h = $(h), d = $(d)", color = palette(:default)[i])
end
plot!(xlims = (0,1))
