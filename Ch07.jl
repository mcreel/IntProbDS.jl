using Plots

##------------------------------------------------------------------------------

# Ch 7.1 Principles of Regression
# Julia code to fit data points using a straight line
N = 50
x = rand(N)
a = 2.5                         # true parameter
b = 1.3                         # true parameter
y = a*x .+ b + 0.2*rand(N)      # Synthesize training data

X = [x ones(N)]                 # construct the X matrix
θ = X\y                         # solve y = X*θ

t    = range(0,stop=1,length=200)
yhat = θ[1]*t .+ θ[2]                  # fitted values at t

p1 = scatter(x,y,makershape=:circle,label="data",legend=:topleft)
plot!(t,yhat,linecolor=:red,linewidth=4,label="best fit")
display(p1)
##------------------------------------------------------------------------------

# Julia code to fit data using a quadratic equation
N = 50
x = rand(N)
a = -2.5
b = 1.3
c = 1.2
X = x.^[0 1 2]                #same as [ones(N) x x.^2]
y = X*[c,b,a] + rand(N)

θ    = X\y
t    = range(0,stop=1,length=200)
yhat = (t.^[0 1 2])*θ               #same as (t.^collect(0:2)')*θ

p2 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:red,linewidth=4,label="fitted curve")
display(p2)
##------------------------------------------------------------------------------

# Julia code to fit data using Legendre polynomials

using LegendrePolynomials

N = 50
x = rand(N)*2 .- 1
a = [-0.001, 0.01, 0.55, 1.5, 1.2]
X = hcat([Pl.(x,p) for p=0:4]...)   # same as [Pl.(x,0) Pl.(x,1) Pl.(x,2) Pl.(x,3) Pl.(x,4)]
y = X*a + 0.5*randn(N)
θ = X\y

t    = range(-1,stop=1,length=200)
yhat = hcat([Pl.(t,p) for p=0:4]...)*θ

p3 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:red,linewidth=4,label="fitted curve")
display(p3)
##------------------------------------------------------------------------------

# Julia code for auto-regressive model
using ToeplitzMatrices

N = 500
y = cumsum(0.2*randn(N)) + 0.05*randn(N)        # generate data

L = 100                                         # use previous 100 samples
c = [0; y[1:400-1]]
r = zeros(L)
X = Matrix(Toeplitz(c,r))                       # Toeplitz matrix, converted
θ = X\y[1:400]                                  # solve y = X*θ
yhat = X*θ                                      # prediction

p4 = scatter(y[1:400],makershape=:circle,label="data",legend=:bottomleft)
plot!(yhat[1:400],linecolor=:red,linewidth=4,label="fitted curve")
display(p4)
##------------------------------------------------------------------------------

# Julia code to demonstrate robust regression

using LinearAlgebra, LegendrePolynomials, Convex, SCS
import MathOptInterface
const MOI = MathOptInterface

#----------------------------------------------------------
"""
    linprog_Convex(c,A,sense,b)

A wrapper for doing linear programming with Convex.jl/SCS.jl.
It solves, for instance, `c'x` st. `A*x<=b`
"""
function linprog_Convex(c,A,sense,b)
    n  = length(c)
    x  = Variable(n)
    #c1 = sense(A*x,b)             #restriction, for Ax <= b, use sense = (<=)
    c1 = A*x <= b
    prob = minimize(dot(c,x),c1)
    solve!(prob,()->SCS.Optimizer(verbose = false))
    x_i = prob.status == MOI.OPTIMAL ? vec(evaluate(x)) : NaN
    return x_i
end
#----------------------------------------------------------

N = 50
x = range(-1,stop=1,length=N)
a = [-0.001, 0.01, 0.55, 1.5, 1.2]
y = hcat([Pl.(x,p) for p=0:4]...)*a  + 0.05*randn(N)        # generate data
idx     = [10, 16, 23, 37, 45]
y[idx] .= 5

X = x.^[0 1 2 3 4]
A = [X -I; -X -I]
b = [y; -y]
c = [zeros(5);ones(N)]

θ = linprog_Convex(c,A,(<=),b)

t    = range(-1,stop=1,length=200)
yhat = (t.^[0 1 2 3 4])*θ[1:5]

p5 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:red,linewidth=4,label="fitted curve")
display(p5)
##------------------------------------------------------------------------------

# Ch 7.2 Overfitting
# Julia: An overfitting example

using LegendrePolynomials

N = 20
x = sort(rand(N)*2 .- 1)
a = [-0.001, 0.01, 0.55, 1.5, 1.2]
y = hcat([Pl.(x,p) for p=0:4]...)*a + 0.1*randn(N)

P = 20
X = hcat([Pl.(x,p) for p=0:P]...)

β = X\y

t    = range(-1,stop=1,length=50)
Xhat = hcat([Pl.(t,p) for p=0:P]...)
yhat = Xhat*β

p6 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft,ylims = (-3,3))
plot!(t,yhat,linecolor=:blue,linewidth=4,label="fitted curve")
display(p6)
##------------------------------------------------------------------------------

using Statistics

Nset = round.( Int,exp10.(range(1,stop=3,length=20)) )  #10,13,16,21,...

E_train = zeros(length(Nset))
E_test  = zeros(length(Nset))
a = [1.3, 2.5]
for j = 1:length(Nset)
    local N,x,E_train_temp,E_test_temp,X
    N = Nset[j]
    x = range(-1,stop=1,length=N)
    E_train_temp = zeros(1000)
    E_test_temp  = zeros(1000)
    X = [ones(N) x]
    for i = 1:1000
        local y,y1,θ,yhat
        y  = a[1] .+ a[2]*x + randn(N)
        y1 = a[1] .+ a[2]*x + randn(N)
        θ = X\y
        yhat = θ[1] .+ θ[2]*x
        E_train_temp[i] = mean(abs2,yhat-y)
        E_test_temp[i]  = mean(abs2,yhat-y1)
    end
    E_train[j] = mean(E_train_temp)
    E_test[j]  = mean(E_test_temp)
end

p7 = scatter(Nset,E_train,xscale=:log10,markershape=:x,markercolor=:black,label="Training Error")
scatter!(Nset,E_test,xscale=:log10,markershape=:circle,markercolor=:red,label="Testing Error")
plot!(Nset,1.0 .- 2.0./Nset,xscale=:log10,linestyle=:solid,color=:black,label="")
plot!(Nset,1.0 .+ 2.0./Nset,xscale=:log10,linestyle=:solid,color=:red,label="")
display(p7)
##------------------------------------------------------------------------------

# Ch 7.3 Bias and Variance
# Julia code to visualize the average predictor

N = 20
a = [5.7, 3.7, -3.6, -2.3, 0.05]
x = range(-1,stop=1,length=N)
X = x.^[0 1 2 3 4]

t    = range(-1,stop=1,length=50)
tMat = t.^[0 1 2 3 4]

yhat = zeros(50,100)                      #50x100 instead of 100x50
for i = 1:100
    local y,θ
    y = X*a + 0.5*randn(N)
    θ = X\y
    yhat[:,i] = tMat*θ
end

p8 = plot(t,yhat,linecolor=:gray,label="")
plot!(t,mean(yhat,dims=2),linecolor=:red,linewidth=4,label="")
display(p8)
##------------------------------------------------------------------------------

# Ch 7.4 Regularization
#  Julia code to demonstrate a ridge regression example

N = 20                                 # Generate data
x = range(-1,stop=1,length=N)
a = [0.5, -2, -3, 4, 6]
y = x.^[0 1 2 3 4]*a + 0.25*randn(N)

λ = 0.1                                # Ridge regression
d = 20
X = x.^collect(0:d-1)'
A = [X; sqrt(λ)I]
b = [y; zeros(d)]
θ = A\b

t    = range(-1,stop=1,length=500)
Xhat = t.^collect(0:d-1)'
yhat = Xhat*θ

p9 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:blue,linewidth=4,label="fitted curve")
display(p9)
##------------------------------------------------------------------------------


#----------------------------------------------------------
"""
    LassoEN(Y,X,λ,δ,β₀)

Do Lasso (set λ>0,δ=0), ridge (set λ=0,δ>0) or elastic net regression (set λ>0,δ>0).
Setting β₀ allows us to specify the target level.

"""
function LassoEN(Y,X,λ,δ=0.0,β₀=0)

    K = size(X,2)
    isa(β₀,Number) && (β₀=fill(β₀,K))

    #b_ls = X\Y                    #LS estimate of weights, no restrictions

    Q  = X'X
    c  = X'Y                      #c'b = Y'X*b

    b  = Variable(K)              #define variables to optimize over
    L1 = quadform(b,Q)            #b'Q*b
    L2 = dot(c,b)                 #c'b
    L3 = norm(b-β₀,1)             #sum(|b-β₀|)
    L4 = sumsquares(b-β₀)         #sum((b-β₀)^2)

    Sol = minimize(L1-2*L2+λ*L3+δ*L4)      #u'u + λ*sum(|b|) + δ*sum(b^2), where u = Y-Xb
    solve!(Sol,()->SCS.Optimizer(verbose = false))
    b_i = Sol.status == MOI.OPTIMAL ? vec(evaluate(b)) : NaN

    return b_i

end
#----------------------------------------------------------

#=
using LinearAlgebra, Convex, SCS
import MathOptInterface
const MOI = MathOptInterface

data = load('./dataset/ch7_data_crime.txt')
y    = data[:,1]        # The observed crime rate
X    = data[:,3:end]    # Feature vectors
(N,d)= size(X)

λset = exp10.(range(-1,stop=8,length=50))    #0.1,0.15,0.23,0.35,...
θ_store = zeros(d,50)
for i = 1:length(λset)
    θ_store[:,i] = LassoEN(y,X,λset[i])
end

p10 = plot(λset,θ_store,xscale=:log10,xlabel="λ",ylabel="eature attribute",
           label = ["funding" "% high" "%no high" "% college" "% graduate"] )
=#
##------------------------------------------------------------------------------

# Julia code to demonstrate overfitting and LASSO

using LinearAlgebra, LegendrePolynomials, Convex, SCS
import MathOptInterface
const MOI = MathOptInterface

N = 20                                          # Generate data
x = range(-1,stop=1,length=N)
a = [1, 0.5, 0.5, 1.5, 1]
y = hcat([Pl.(x,p) for p=0:4]...)*a + 0.25*randn(N)

d = 20                                          # Solve LASSO using Convex.jl
X = hcat([Pl.(x,p) for p=0:d-1]...)
λ = 2
θ = LassoEN(y,X,λ)

t    = range(-1,stop=1,length=200)
Xhat = hcat([Pl.(t,p) for p=0:d-1]...)
yhat = Xhat*θ

p11 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:blue,linewidth=4,label="fitted curve")
display(p11)
#------------------------------------------------------------------------------
