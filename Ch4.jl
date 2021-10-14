# Chapter 4.3

# CDF of a Gaussian mixture

## Julia code to generate the PDF and CDF
using Distributions: Normal, MixtureModel
using Plots

gmm = MixtureModel(Normal[Normal(0, 1), Normal(5, 1),], [0.3, 0.7])

p1 = plot(x -> 0, -5, 1, fillrange= x->pdf(gmm, x), 
    linealpha=0, fillcolor=RGB(0.8, 0.8, 1), alpha=0.5, label=false) # style
plot!(p1, x -> pdf(gmm, x), -5, 10, 
    color=1, linewidth=6, label="PDF") # style

p2 = plot(x -> cdf(gmm, x), -5, 10, 
    linewidth=6, label="CDF")

plot(p1, p2, layout=(1, 2), 
    size=(600, 300), legend=:topleft)



# CDF of a uniform random variable
u = Uniform(-3, 4)

p1 = plot(x -> 0, -5:0.01:1, fillrange=x->pdf(u, x),
    linealpha=0, fillcolor=RGB(0.8, 0.8, 1), alpha=0.5, label=false)
plot!(p1, x -> pdf(u, x), -5, 10, 
    color=1, linewidth=6, ylims=(0, 0.4), label="PDF")

p2 = plot(x -> cdf(u, x), -5:0.01:10, 
    linewidth=6, label="CDF")
vline!(p2, [-3, 4], 
    linestyle=:dash, color=:green, label=false)

plot(p1, p2, layout=(1, 2), 
    size=(600, 300), legend=:topleft)