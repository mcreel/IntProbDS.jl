# Julia code to compute the correlation coefficient
using Distributions
using Plots

x = rand(MvNormal([0, 0], [3 1; 1 1]), 1000)
# x = rand(MvNormal([0, 0], [3 0; 0 3]), 1000)
# x = rand(MvNormal([0, 0], [3 2.9; 2.9 3]), 1000)
scatter(x[1, :], x[2, :])

σ₁ = std(x[1, :])
σ₂ = std(x[2, :])
μ₁ = mean(x[1, :])
μ₂ = mean(x[2, :])
Exy = mean(x[1, :] .* x[2, :])
ρ = (Exy - μ₁ * μ₂) / (σ₁ * σ₂)

####################################################################

# Julia code to compute a mean vector.
using Statistics

X = randn(100, 2)
mean(X, dims=1)

####################################################################

# Julia code to compute covariance matrix.
using Statistics

X = randn(100, 2)
cov(X)

####################################################################

# Julia code: Overlay random numbers with the Gaussian contour.
using Distributions
using Plots

p = MvNormal([0, 0], [0.25 0.3; 0.3 1])

X = rand(p, 1000)
x₁ = -2.5:0.01:2.5
x₂ = -3.5:0.01:3.5

f(x₁, x₂) = pdf(p, [x₁, x₂])

scatter(X[1, :], X[2, :], legend=false)
contour!(x₁, x₂, f, linewidth=2)

####################################################################

# Julia code: Gaussian(0,1) --> Gaussian(mu,sigma)
using Distributions

x = rand(MvNormal([0, 0], [1 0; 0 1]), 1000)
Σ = [3 -0.5; -0.5 1]
μ = [1, -2]
y = Σ^(1/2) * x .+ μ

####################################################################

# Julia code: Gaussian(mu,sigma) --> Gaussian(0,1)
using Distributions

y = rand(MvNormal([1, -2], [3 -0.5; -0.5 1]), 100)
μ = mean(y, dims=2)
Σ = cov(y')
x = Σ^(-1/2) * (y .- μ)

####################################################################

# Julia code to perform the principal component analysis
using LinearAlgebra
using Distributions

x = rand(MvNormal([0, 0], [2 -1.9; -1.9 2]), 1000)
Σ = cov(x')
S, U = eigen(Σ)
U[:, 1]
U[:, 2]
