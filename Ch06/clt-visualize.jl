# Julia: Visualize the Central Limit Theorem
using Distributions, Plots

N = 10
x = range(0, N, length=1001)
p = 0.5

p_b(x) = pdf(Binomial(N, p), x)
p_n(x) = pdf(Normal(N*p, sqrt(N*p*(1-p))), x)

c_b(x) = cdf(Binomial(N, p), x)
c_n(x) = cdf(Normal(N*p, sqrt(N*p*(1-p))), x)


xs = range(5-2.5, 5 + 2.5, length=1001)


p1 = plot(xs, p_n.(xs); fillrange=0 .* xs, color=RGB(0.6, 0.9, 1.0), legend=false,
     title="Binomial PDF")
plot!(0:N, p_b.(0:N); color=RGB(0,0,0), linewidth=2, seriestype=:sticks, layout=1)


p2 = plot(xs, p_n.(xs), fillrange=0 .* xs, color=RGB(0.6, 0.9, 1.0), legend=false,
     title = "Gaussian PDF")
plot!(p_n, 0, N; color=RGB(0.8, 0, 0), linewidth=6, layout=2)

l = @layout [a b]
plot(p1, p2, layout=l)
