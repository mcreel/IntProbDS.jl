# Julia: Central Limit Theorem from moment generating functions
using Distributions, Plots

N = 2
p = 0.5;

B = Bernoulli(p)
μ, σ = mean(B), std(B)/sqrt(N) # for Xbar
Z = Normal(μ, σ)

m_Bernoulli(s) = mgf(B,s)    # (1 - p + p*exp(s))
m_B(s) = m_Bernoulli(s/N)^N  # (X₁ + ⋯ + Xₙ)/n
m_Z(s) = mgf(Z,s)            # exp(μ * s + σ^2*s^2/2)

bin_args = (linewidth=8, color=RGB(0.1, 0.6, 1),
            yaxis=:log, yticks = [10.0^i for i in -1:4],
            label="Binomial MGF")
norm_args = (linewidth=8, color=RGB(0, 0, 0), yaxis=:log, linestyle=:dot, label="Gaussian MGF")

plot(m_B, -10, 10; bin_args..., legend=:topleft)
plot!(m_Z; norm_args...)
