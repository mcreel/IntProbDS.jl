# Julia code to compare the probability bounds
using Distributions, Plots

ϵ = 0.1
σ   = 1

p_exact(N) = 1 - cdf(Normal(), sqrt(N) * ϵ / σ)
p_cheby(N) = σ^2 / (ϵ^2 * N);
p_chern(N) = exp(-ϵ^2 * N / (2*σ^2));

ex_args    = (linewidth=2, color=RGB(1.0, 0.5, 0.0), label="Exact", markershape=:circle)
chb_args   = (linewidth=2, color=RGB(0.2, 0.7, 0.1), label="Chebyshev", linestyle=:dashdot)
chern_args = (linewidth=2, color=RGB(0.2, 0.0, 0.8), label="Chernoff")

Ns = [10^i for i in range(1, 3.8, length=50)]
scatter(Ns, p_exact.(Ns);  ex_args...,
        xaxis=:log10,
        yaxis=:log10, yticks=[1e-15, 1e-10, 1e-5, 1e0],
        legend=:bottomleft)
plot!(p_cheby; chb_args...)
plot!(p_chern; chern_args...)
