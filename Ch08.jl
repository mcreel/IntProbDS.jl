# Chapter 8.1 Maximum-likelihood Estimation
## Visualizing the likelihood function
using Plots
N = 50
S = range(1., N, step=0.1)
θ = range(0.1, 0.9,length=100)
L(S, θ) = S*log(θ) + (N-S)*log(1. - θ)
## the surface plot
plotlyjs() # this backend allows rotating with mouse
p1 = surface(S, θ, (S, θ) -> L(S, θ), color=:jet, xlabel="S", ylabel="θ",
    title="ℒ(θ|S)") 
## the bird's eye view
gr()
p2 = heatmap(S, θ, (S, θ) -> L(S, θ), color=:jet, xlabel="S", ylabel="θ",
    title="bird's eye view")
SS = 25 # set the value for the slice here
vline!([SS], label=false, color=:black)
## slice through the log likelihood 
p3 = plot(θ, θ -> L(SS, θ),label=false, xlabel="θ", title="ℒ(θ|S=$SS)") 
plot(p2, p3)

## ML estimation for single-photon imaging
using Images, Distributions
using ImageView:imshow
download("https://probability4datascience.com/data/cameraman.tif", "./cameraman.tif")
λ  = Float64.(load("cameraman.tif"))
T = 100
x = [rand.(Poisson.(λ)) for _=1:T]
y = [x[i] .>= 1. for i=1:T]
λhat = -log.(1. .- mean(y))
imshow(x[1]) # a single sample image
imshow(λhat) # the ML recovered image


## Chapter 8.2 Properties of the ML estimation
# Visualizing the invariance principle
using Plots
N = 50
S = 20
θ = range(0.1,0.9,length=1000)
η  = -log.(1. .- θ) # the reparameterization transformation
L₁(θ) = S*log(θ) + (N-S)*log(1. - θ) # log likelihood in terms of θ 
L₂(η) = S*log(1. - exp(-η)) - (N-S)*η # log likelihood in terms of η 
## plot the log likelihood with θ parameterization 
p4 = plot(θ, θ -> L₁(θ), linewidth=5, color=:blue, label=false)
xlabel!("θ")
title!("ℒ₁(θ|S=$S)") 
vline!([0.4], label=false, color=:red)
## plot the transformation
p2 = plot(θ, η, linewidth=5, color=:black, label=false)
xlabel!("θ")
ylabel!("η") 
title!("η(θ)") 
vline!([0.4], label=false, color=:red)
hline!([0.5], label=false, color=:green)
## plot the log likelihood with η parameterization
p1 = plot(L₂.(η), η,  linewidth=5, color=:blue, label=false, xflip=true)
ylabel!("η") 
hline!([0.5], label=false, color=:green)
title!("ℒ₂(η|S=$S)")
## blank place holder plot
p3 = plot(grid=false, xaxis=false, yaxis=false, xticks=false, yticks=false)
# plot the whole set in the final image
plot(p1, p2, p3, p4)


## Chapter 8.3 Maximum-a-Posteriori Estimation
# Influence of the priors
# likelihood is N(θ,1) 
using Plots, Distributions
N = 5       # sample size
μ₀  = 0.0   # prior mean
σ₀  = 1.    # prior std. dev.
θ = 5.      # true mean of sample data
x = rand(Normal(θ,1), N)    # the sample data
xbar  = mean(x) # mean of sample data
t = range(-3.,7.,length=1000)
θhat = (σ₀^2. * xbar + μ₀/N)/(σ₀^2. + 1. /N) # MAP
σhat = sqrt(1. / (1. / σ₀^2. + N))  # s.d. posterior 
p0 = pdf.(Normal(xbar,1.0), t)       # likelihood
p1 = pdf.(Normal(θhat, σhat), t)  # posterior
prior = pdf.(Normal(μ₀,σ₀),t)     # prior, divided by 10 to 
plot(t, [p0, p1, prior], label=["Likelihood" "Posterior" "prior"], linewidth=3, legend=:topleft)
plot!([x], [0.1*ones(N)], line=:stem, label="data", marker=:circle, linewidth=3, title="N = $N")


## Conjugate priors
using Plots, Distributions
σ₀ = 0.25
μ₀ = 0.0
μ  = 1.
σ  = 0.25
Nset = [0 1 2 5 8 12 20]
x0 = σ .* randn(100)
posterior = zeros(1000,7)
t = range(-1.,1.5,length=1000)
p0 = pdf.(Normal(0.,1.), t)
for i=1:7
    N = Nset[i]
    x = x0[1:N]
    θ  = μ * (N*σ₀^2.)/(N*σ₀^2. + σ ^2.) + μ₀*(σ^2.)/(N*σ₀^2. + σ^2.)
    σN= sqrt(1. /(1. /σ₀^2. + N/σ^2.))
    posterior[:,i] = pdf.(Normal(θ, σN), t)
end
plot(t, posterior, linewidth=2., 
    label=["N = 0" "N = 1" "N = 2" "N = 5" "N = 8" "N = 12" "N = 20"],
    legend=:topleft)


