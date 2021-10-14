# Chapter 4.3

# CDF of a Gaussian mixture

## Julia code to generate the PDF and CDF
using Distributions: Normal, MixtureModel
using Plots

gmm = MixtureModel(Normal[Normal(0, 1), Normal(5, 1),], [0.3, 0.7])

# x = range(-5, 10, length=1000)
# f = pdf(gmm, x)

# x1 = range(-5, 1, length=1000)
# f1 = pdf(gmm, x1)

# p1 = plot(x1, zeros(1000), fillrange=f1, 
#     linealpha=0, fillcolor=RGB(0.8, 0.8, 1), alpha=0.5, label=false)
# plot!(p1, x -> pdf(gmm, x), -5, 10, label="PDF", linewidth=6, color=1)

# F = cdf(gmm, x)
# p2 = plot(x, F, linewidth=6, label="CDF")

p1 = plot(x -> 0, -5, 1, fillrange= x->pdf(gmm, x), 
    linealpha=0, fillcolor=RGB(0.8, 0.8, 1), alpha=0.5, label=false) # style
plot!(p1, x -> pdf(gmm, x), -5, 10, 
    color=1, linewidth=6, label="PDF") # style

p2 = plot(x -> cdf(gmm, x), -5, 10, 
    linewidth=6, label="CDF")

plot(p1, p2, layout=(1, 2), 
    size=(600, 300), legend=:topleft)
