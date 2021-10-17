# Julia code for Chapter 9

# Chapter 9.1 Confidence Interval

####################################################################
## Histogram of the sample average
using Distributions, Plots, Printf
gmm = MixtureModel(Normal[Normal(), Normal(5., 1.),], [0.3, 0.7])
x = range(-5., 10., length=1000)
# the population distribution
plot(x, pdf(gmm, x), linewidth=3, label=false, title="Population distribution")
# histograms of 4 random samples of size 50
plot_array = [] 
for i=1:4
    bin = -5:10
    Y = rand(gmm, 50)
    tt = @sprintf("Mean = %3.2f", mean(Y))
    push!(plot_array, histogram(Y, bins=bin, label = tt, legend=:topleft))
end
plot(plot_array...)
# histogram of sample average
M = zeros(10000)
bin = range(2., 5., step=0.1)
for i = 1:10000
    M[i] = mean(rand(gmm, 50))
end   
histogram(M, bins=bin, label=false, title="Histogram of the sample average")

####################################################################
## Compute confidence interval
# Julia code to compute the width of the confidence interval
using Distributions
α = 0.05
μ = 0.
σ = 1.
ϵ = quantile(Normal(μ, σ ), 1. - α/2.) # CI is estimator ± ϵ

####################################################################
## Visualize the t-distribution
# Julia code to plot the t-distribution
using Distributions, Plots
x = range(-5.,5.,length=100)
p1 = pdf(Normal(),x)
p2 = pdf(TDist(11-1),x)
p3 = pdf(TDist(3-1),x)
p4 = pdf(TDist(2-1),x)
plot([p1,p2,p3,p4], label = ["Gaussian(0,1)" "t-dist, N = 11" "t-dist, N = 3" "t-dist, N = 2"], legend=:topright)

####################################################################
## Construct a confidence interval from data
# Julia code to generate a confidence interval
using Statistics, Distributions, Printf
x = [72., 69., 75., 58., 67., 70., 60., 71., 59., 65.]
θhat = mean(x)  # Sample mean
σhat = std(x)   # Sample standard deviation
N = size(x,1)
ν = N-1 # degrees of freedom
α = 0.05        # confidence level
z = quantile(TDist(ν), 1. - α/2.)
CI_L = θhat - z*σhat/sqrt(N)
CI_U = θhat + z*σhat/sqrt(N)
@printf("Confidence interval: [%5.3f - %5.3f]", CI_L, CI_U)


####################################################################

# Chapter 9.2 Bootstrap

####################################################################
## Visualize bootstrap
# Julia code to illustrate bootstrap
using Plots, Distributions, Statistics
gmm = MixtureModel(Normal[Normal(), Normal(5., 1.),], [0.3, 0.7])
# visualize the population
x = range(-5., 10., length=1000);
f = pdf(gmm, x);
plot(x, pdf(gmm,x), linewidth=3, label="Population", legend=:topleft)
# Visualize bootstrap dataset, the real sample data
N = 100
X = rand(gmm, N) # the real sample data
p1 = histogram(X, bins = range(-5,10,length=30), label=false)
# visualize 3 bootstrap samples
plot_array = [] 
for i=1:3
    # Bootstrap dataset
    idx = rand(1:N, N)
    Xb1 = X[idx]
    push!(plot_array, histogram(Xb1, bins = range(-5.,10.,length=30)))
end
plot(plot_array..., label=false)

# Bootstrap the median
K = 1000 # number of bootstrap re-samples
θhat = zeros(K)
for i=1:K               # repeat K times
    idx = rand(1:N, N)  # sampling w/ replacement
    Y = X[idx]
    θhat[i] = median(Y) # estimator
end
M = mean(θhat)          # bootstrap mean
V = var(θhat)           # bootstrap variance
# histogram of bootstrap replications of estimator
histogram(θhat, bins=range(3.5,5.5,length=30), legend=false)

####################################################################
## Bootstrap median using real data
# Julia code to estimate a bootstrap variance
using Statistics
X = [72., 69., 75., 58., 67., 70., 60., 71., 59., 65.]
N = size(X,1)
K = 1000
θhat = zeros(K)
for i=1:K                       # repeat K times
    idx = rand(1:N, N)          # sampling w/ replacement
    Y = X[idx]
    θhat[i] = median(Y)         # estimator
end
M = mean(θhat)                  # bootstrap mean
V = var(θhat)                   # bootstrap variance

####################################################################

# Chapter 9.3 Hypothesis Testing

####################################################################
## Estimate Z-value
θhat = 0.29                 # Your estimate
θ = 0.25                    # Your hypothesis
σ = sqrt(θ*(1. - θ))        # Known standard deviation
N = 1000                    # Number of samples
Z_hat = (θhat - θ)/(σ/sqrt(N))

####################################################################
## Compute critical value
using Distributions
α = 0.05
z_α = quantile(Normal(), 1-α)

####################################################################
## Compute p-value
using Distributions
p = cdf(Normal(), -1.92)

####################################################################

# Chapter 9.5 ROC Curve

####################################################################
## Plot an ROC curve
# Julia code to plot ROC curve
using Distributions, Plots, Statistics
σ = 2.
μ = 3.
PF1 = zeros(1000); PD1 = zeros(1000)
PF2 = zeros(1000); PD2 = zeros(1000)
αset = range(0.,1.,length=1000)
d = Normal()
for i=1:1000
    α = αset[i]
    PF1[i] = α
    PD1[i] = α
    PF2[i] = α
    PD2[i] = 1. - cdf(d, quantile(d, 1-α)-μ/σ)
end
plot(PF1, PD1, linewidth=3, label = "Blind guess")
plot!(PF2, PD2, linewidth=3, label = "Neyman-Pearson", legend=:bottomright)
# Computer area under curve
auc1 = sum(PD1.*[0.; diff(PF1)])
auc2 = sum(PD2.*[0.; diff(PF2)])


####################################################################
## Another ROC curve
# Julia code to generate the ROC curve.
using Distributions, Plots
σ = 2.
μ = 3.
PF1 = zeros(1000); PD1 = zeros(1000)
PF2 = zeros(1000); PD2 = zeros(1000)
αset = range(0.,0.5,length=1000)
d = Normal()
for i=1:1000
    α = αset[i]
    PF1[i] = 2. *α
    PD1[i] = 1. - (cdf(d, quantile(d,1. - α) - μ/σ) - 
        cdf(d, -quantile(d, 1. -α) - μ/σ))
end
αset = range(0.,1.,length=1000)
for i=1:1000
    α = αset[i]
    PF2[i] = α
    PD2[i] = 1. - cdf(d, quantile(d, 1-α)-μ/σ)
end
args = (linewidth=3, xlabel = "p_F", ylabel="p_D")
plot(PF1, PD1; args..., label = "Proposed decision")
plot!(PF2, PD2; args..., label = "Neyman-Pearson", legend=:bottomright)

####################################################################
## ROC on real data
using DelimitedFiles, Plots
scores = readdlm("ch9_ROC_example_data.txt")
scores = scores[:] # a vector
tau = range(0.,1.,length=1000)
labels = [ones(50); zeros(50)]
PF = zeros(1000)
PD = zeros(1000)
for i=1:1000
    idx   = (scores .<= tau[i])
    predict = zeros(100)
    predict[idx]   .= 1.
    true_positive  = 0; true_negative  = 0;
    false_positive = 0; false_negative = 0;
    for j=1:100
        if (predict[j]==1) && (labels[j]==1)
            true_positive += 1   end
        if (predict[j]==1) && (labels[j]==0)
            false_positive += 1 end
        if (predict[j]==0) && (labels[j]==1)
            false_negative += 1 end
        if (predict[j]==0) && (labels[j]==0)
            true_negative += 1   end
    end
    PF[i] = false_positive/50
    PD[i] = true_positive/50
end
plot(PF, PD, linewidth=3, legend=false)
