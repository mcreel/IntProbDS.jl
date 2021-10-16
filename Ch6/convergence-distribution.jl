# Julia: Visualize convergence in distribution
using Distributions, Plots
N, p = 10, 0.5

B = Binomial(N,p)
Z = Normal(mean(B), std(B)) # N*p, sqrt(N*p*(1-p))

p_b(x) = pdf(B, x)
p_n(x) = pdf(Z, x)

c_b(x) = cdf(B, x)
c_n(x) = cdf(Z, x)

bin_args = (color = RGB(0, 0, 0), linewidth=2, label="Binomial")
norm_args = (color = RGB(0.8, 0, 0), linewidth=6, label="Normal")

ns = 0:N
p1 = plot(ns, p_b.(ns); bin_args..., seriestype=:sticks)
plot!(xs, p_n; norm_args...)

p2 = plot(xs, c_b; bin_args...)
plot!(xs, c_n; norm_args...)

l = @layout [a b]
plot(p1, p2, layout=l)
