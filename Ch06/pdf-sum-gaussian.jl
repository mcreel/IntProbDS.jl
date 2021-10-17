# Julia: Plot the PDF of the sum of two Gaussians
using Distributions, Plots

n = 10000;

K = 2 # 6

X = DiscreteUniform(1, 6)
μ = mean(X)   # 3.5
σ = std(X)    # sqrt((6^2-1)/12)

Z = sum(rand(X, n) for _ in 1:K)


mk = K * μ
sk = sqrt(K) * σ

bin_range = (floor(mk - 3sk) - 1/2):(ceil(mk + 3sk) + 1/2)
plot_args = (normalize=true, color=RGB(0, 0.5, 0.8), linewidth=2, legend=false,
             bins=bin_range)

histogram(Z; plot_args..., title="⚁"^K)
