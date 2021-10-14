# Chapter 1.1

# Visualizing a geometric series

## Julia code to generate a geometric series
using Plots:bar
p = 0.5
n = 1:10
X = p .^ n
bar(n, X, legend=false)

## compute N choose K
# Julia code to (N choose K) and K!
n = 10
k = 2
binomial(n, k)
factorial(k)

# Chapter 1.4

# Inner product of two vectors

## Julia code to perform and inner product
x = [1, 0, -1]
y = [3, 2, 0]
x'y

## Norm of a vector
# Julia code to compute the norm
using LinearAlgebra:norm
x = [1, 0, -1]
norm(x)

## Weighted norm of a vector
# Julia code to compute the weighted norm
W = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
x = [2, -1, 1]
z = x'W*x

## System of linear equations
# Julia code to solve Xβ = y
X = [1. 3.; -2. 7.; 0. 1.]
y = [2., 1., 0.]
β = X\y
