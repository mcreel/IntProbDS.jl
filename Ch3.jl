# Exercise "Histogram of the English alphabets"
# Julia code generate the histogram
using Plots,DelimitedFiles
f = readdlm("ch3_data_english.txt")
bar(f/100,xticks = (1:26,'a':'z'),yticks = 0.0:0.02:0.1,label = false)

# Exercise "Histogram of the throwing a die"
# Julia code generate the histogram
using Plots,Random
q = rand(1:6,100)
histogram(q,bins = 6,normalize = true,label = "N = 100")

# Exercise "Histogram of an exponential random variable"
# Julia code used to generate the plots
using Plots,Random,Distributions
λ = 1
X = rand(Exponential(1/λ),1_000)
histogram(X,bins = 200,label = "K = 200")

# Exercise "Cross validation loss"
using Plots,Random,Distributions,StatsBase
λ = 1
n = 1_000
X = rand(Exponential(1/λ),n)
m = 6:200
J = zeros(195)
for i = 1:195
    hist = fit(Histogram,X,range(minimum(X),maximum(X),length = m[i]+1))
    h = n/m[i]
    J[i] = 2/((n-1)*h)-((n+1)/((n-1)*h))*sum( (hist.weights/n).^2 )
end

plot(m,J,label = false)

# Exercise "Expectation E[X] where X is uniform from [0,1]"
using Random,Statistics
X = rand(10_000)
mean(X)

# Exercise "Mean from PMF"
# Julia code to compute the expectation
p = [0.25,0.5,0.25]
x = [0,1,2]
EX = sum(p .* x)

# Exercise "Mean of geometric random variable"
k = 1:100
p = 0.5.^k
EX = sum(p .* k)

# Exercise "Bernoulli Random Variables"
using Plots,Distributions
p = 0.5
n = 1
X = rand(Binomial(n,p),1_000)
histogram(X,bins = 2,label = false)

# Julia code to generate Erdos Renyi Graph
using Plots,Random,LinearAlgebra,GraphRecipes
A = rand(40,40) .< 0.1
A = triu(A,1)
A = A + A'
graphplot(A)

# Exercise "Binomial Random Variables" 2
using Plots,Distributions
p = 0.5
n = 10
X = rand(Binomial(n,p),5_000)
histogram(X,label = false)

# Exercise "Binomial CDF"
using Plots,Distributions
x = 0:10
p = 0.5
n = 10
F = cdf.(Binomial(n,p),x)
plot(x,F,linetype = :steppost,label = false,markershape = :circle)

# Exercise "Poisson CDF"
## Code for the Poisson PMF 
using Plots,Distributions
λ_set = [1,4,10]
p = zeros(21,3)
k = 0:1:20
for i = 1:3
    p[:,i] = pdf.(Poisson(λ_set[i]),k)
end
plot(k,p[:,1],line = :stem,markershape = :circle,label = "λ = 1")
plot!(k,p[:,2],line = :stem,markershape = :circle,label = "λ = 4")
plot!(k,p[:,3],line = :stem,markershape = :circle,label = "λ = 10")

## Code for Poisson CDF
using Plots,Distributions
λ_set = [1,4,10]
p = zeros(21,3)
k = 0:1:20
for i = 1:3
    p[:,i] = cdf.(Poisson(λ_set[i]),k)
end
plot(k,p[:,1],line = :steppost,markershape = :circle,label = "λ = 1")
plot!(k,p[:,2],line = :steppost,markershape = :circle,label = "λ = 4")
plot!(k,p[:,3],line = :steppost,markershape = :circle,label = "λ = 10",legend = :bottomright)

# Exercise "Poisson-Binomial approximation"
using Plots,Distributions
n = 5_000
p = 0.01
λ = n*p
x = 0:120
y = pdf.(Binomial(n,p),x)
z = pdf.(Poisson(λ),x)
plot(x,y,line = :stem,markershape = :circle,label = "Binomial,n = 5000,p = 0.01")
plot!(x,z,label = "Poisson,λ = 50")

# Photon Shot Noise
using Distributions,Images,ImageView,Random
x = Float64.(load("cameraman.tif"))

α = 10
λ = α*x
X1 = rand.(Poisson.(λ))
imshow(X1)

α = 100
λ = α*x
X1 = rand.(Poisson.(λ))
imshow(X1)

α = 1_000
λ = α*x
X1 = rand.(Poisson.(λ))
imshow(X1)
