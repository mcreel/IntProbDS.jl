println("Chapter 1.1")
println("visualizing a geometric series")
using StatsPlots
p = 0.5
n = 1:10
X = p .^ n
bar(n, X, legend=false)

println("compute N choose K")
n = 10
k = 2
@show binomial(n, k)
@show factorial(k)

println("Chapter 1.4")
println("Inner product of two vectors")
using LinearAlgebra
x = [1, 0, -1]
y = [3, 2, 0]
@show dot(x,y)
println("Norm of a vector")
x = [1, 0, -1]
@show norm(x)
nothing
