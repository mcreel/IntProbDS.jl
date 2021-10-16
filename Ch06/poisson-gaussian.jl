# Julia: Poisson to Gaussian: convergence in distribution
using Distributions, Plots

N = 4; # N = 10, N = 50;
lambda = 1

P = Poisson(N*lambda)
Z = Normal(mean(P), std(P))

p_b(x) = pdf(P, x) # evaluate on integers
p_n(x) = pdf(Z, x)

c_b(x) = cdf(P, x)
c_n(x) = cdf(Z, x)

bin_args = (linewidth=2, color=RGB(0.0, 0.0, 0.0), label="Poisson")
norm_args = (linewidth=6, colr=RGB(0.8, 0.0, 0.0), label="Gaussian")

ns = 0:2N
p1 = plot(ns, p_b.(ns); bin_args..., seriestype=:sticks, legend=:topright)
plot!(p_n, 0, 2N; norm_args...)

p2 = plot(c_b, 0, 2N; bin_args..., legend=:bottomright)
plot!(c_n; norm_args...)

l = @layout [a b]
plot(p1, p2, layout=l)
