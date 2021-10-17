# Julia: Visualize the Central Limit Theorem
using Distributions, Plots

N = 10
x = range(0, N, length=1001)
p = 0.5


B = Binomial(N,p)
Z = Normal(mean(B), std(B))  # N*p, sqrt(N*p*(1-p))

p_b(x) = pdf(B, x)
p_n(x) = pdf(Z, x)

c_b(x) = cdf(B, x)
c_n(x) = cdf(Z, x)


xs = range(5-2.5, 5 + 2.5, length=1001)

p1 = plot(p_n, 5-2.5, 5+2.5; fillrange=0 .* xs, color=RGB(0.6, 0.9, 1.0), legend=false,
     title="Binomial PDF")
plot!(0:N, p_b.(0:N); color=RGB(0,0,0), linewidth=2, seriestype=:sticks, layout=1)


p2 = plot(p_n, 5-2.5, 5+2.5, fillrange=0 .* xs, color=RGB(0.6, 0.9, 1.0), legend=false,
     title = "Gaussian PDF")
plot!(p_n, 0, N; color=RGB(0.8, 0, 0), linewidth=6, layout=2)

l = @layout [a b]
plot(p1, p2, layout=l)
