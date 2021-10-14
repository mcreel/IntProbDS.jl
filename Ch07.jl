using Plots

##------------------------------------------------------------------------------

# Julia code to fit data points using a straight line
N = 50
x = rand(N)
a = 2.5                         # true parameter
b = 1.3                         # true parameter
y = a*x .+ b + 0.2*rand(N)      # Synthesize training data

X    = [x ones(N)]              # construct the X matrix
theta = X\y                     # solve y = X theta

t    = range(0, stop=1, length=200)    # interpolate and plot
yhat = theta[1]*t .+ theta[2]

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
y = a*x.^2 + b*x .+ c + 1*rand(N)

X     = [ones(N) x x.^2];
theta = X\y
t     = range(0, stop=1, length=200);
yhat  = theta[3]*t.^2 + theta[2]*t .+ theta[1]

p2 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:red,linewidth=4,label="fitted curve")
display(p2)
##------------------------------------------------------------------------------

# Julia code to fit data using Legendre polynomials

using LegendrePolynomials

N = 50
x = rand(N)*2 .- 1
a = [-0.001, 0.01, 0.55, 1.5, 1.2]
y = a[1]*Pl.(x,0) + a[2]*Pl.(x,1) +
    a[3]*Pl.(x,2) + a[4]*Pl.(x,3) +
    a[5]*Pl.(x,4) + 0.5*randn(N)

X = [Pl.(x,0) Pl.(x,1) Pl.(x,2) Pl.(x,3) Pl.(x,4)]
theta = X\y

t    = range(-1, stop=1, length=200)
yhat = theta[1]*Pl.(t,0) + theta[2]*Pl.(t,1) +
       theta[3]*Pl.(t,2) + theta[4]*Pl.(t,3) +
       theta[5]*Pl.(t,4)

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
theta = X\y[1:400]                              # solve y = X theta
yhat  = X*theta                                 # prediction

p4 = scatter(y[1:400],makershape=:circle,label="data",legend=:bottomleft)
plot!(yhat[1:400],linecolor=:red,linewidth=4,label="fitted curve")
display(p4)
##------------------------------------------------------------------------------

# Julia code to demonstrate robust regression
using LinearAlgebra, MathProgBase, Ipopt

N = 50
x = range(-1,stop=1,length=N)
a = [-0.001, 0.01, 0.55, 1.5, 1.2]
y = a[1]*Pl.(x,0) + a[2]*Pl.(x,1) +
    a[3]*Pl.(x,2) + a[4]*Pl.(x,3) +
    a[5]*Pl.(x,4) + 0.2*randn(N)
idx  = [10, 16, 23, 37, 45]
y[idx] .= 5

X    = [x.^0 x.^1 x.^2 x.^3 x.^4]
A    = [X -I; -X -I]
b    = [y; -y]
c    = [zeros(5);ones(N)]

Sol = linprog(c,A,'<',b,IpoptSolver(print_level=0))
if Sol.status == :Optimal
  theta = Sol.sol
end

t    = range(-1,stop=1,length=200)
yhat = theta[1]*Pl.(t,0) + theta[2]*Pl.(t,1) +
       theta[3]*Pl.(t,2) + theta[4]*Pl.(t,3) +
       theta[5]*Pl.(t,4)

p5 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:red,linewidth=4,label="fitted curve")
display(p5)
##------------------------------------------------------------------------------

# Julia: An overfitting example
N = 20
x = sort(rand(N)*2 .- 1)
a = [-0.001, 0.01, 0.55, 1.5, 1.2];
y = a[1]*Pl.(x,0) + a[2]*Pl.(x,1) +
    a[3]*Pl.(x,2) + a[4]*Pl.(x,3) +
    a[5]*Pl.(x,4) + 0.1*randn(N)


P = 20;
X = zeros(N,P+1)
for p = 0:P
    X[:,p+1] = Pl.(x,p)
end
beta = X\y

t    = range(-1, stop=1, length=50);
Xhat = zeros(length(t),P+1)
for p = 0:P
    Xhat[:,p+1] = Pl.(t,p)
end
yhat = Xhat*beta

p6 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft,ylims = (-3,3))
plot!(t,yhat,linecolor=:blue,linewidth=4,label="fitted curve")
display(p6)
#------------------------------------------------------------------------------

using Statistics

Nset = round.(Int,exp10.(range(1,stop=3,length=20)) )

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
        local y,y1,theta,yhat
        y  = a[1] .+ a[2]*x + randn(size(x))
        y1 = a[1] .+ a[2]*x + randn(size(x))
        theta = X\y
        yhat = theta[1] .+ theta[2]*x
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
#------------------------------------------------------------------------------

# Julia code to visualize the average predictor
N = 20
a = [5.7, 3.7, -3.6, -2.3, 0.05]
x = range(-1, stop=1,length=N)
X = [x.^0 x.^1 x.^2 x.^3 x.^4]
t = range(-1, stop=1, length=50)

yhat = zeros(50,100)                      #50x100 instead of 100x50
for i = 1:100
    local y,theta
    y     = X*a + 0.5*randn(N)
    theta = X\y
    yhat[:,i] = theta[1] .+ theta[2]*t + theta[3]*t.^2 + theta[4]*t.^3 + theta[5]*t.^4
end

p8 = plot(t,yhat,linecolor=:gray,label="")
plot!(t,mean(yhat,dims=2),linecolor=:red,linewidth=4,label="")
display(p8)
#------------------------------------------------------------------------------

#  Julia code to demonstrate a ridge regression example
# Generate data
N = 20
x = range(-1, stop=1,length=N)
a = [0.5, -2, -3, 4, 6]
y = a[1] .+ a[2]*x + a[3]*x.^2 + a[4]*x.^3 + a[5]*x.^4 + 0.25*randn(N)

# Ridge regression
lambda = 0.1
d = 20
X = zeros(N,d)
for p = 0:d-1
    X[:,p+1] = x.^p
end
A = [X; sqrt(lambda)I]
b = [y; zeros(d,1)]
theta = A\b

# Interpolate and display results
t    = range(-1, stop=1, length=500)
Xhat = zeros(length(t),d)
for p = 0:d-1
    Xhat[:,p+1] = t.^p
end

yhat = Xhat*theta

p9 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:blue,linewidth=4,label="fitted curve")
display(p9)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------

using Convex, SCS

import MathOptInterface
const MOI = MathOptInterface

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

#------------------------------------------------------------------------------
#=
data = load('./dataset/ch7_data_crime.txt')
y    = data[:,1]        # The observed crime rate
X    = data[:,3:end]    # Feature vectors
(N,d)= size(X)

lambdaset = exp10.(range(-1,stop=8,length=50))
theta_store   = zeros(d,50)
for i = 1:length(lambdaset)
    theta_store[:,i] = LassoEN(y,X,lambdaset[i])
end

p10 = plot(lambdaset,theta_store,xscale=:log10,xlabel="lambda",ylabel="eature attribute",
           label = ["funding" "% high" "%no high" "% college" "% graduate"] )
=#
#------------------------------------------------------------------------------

# Julia code to demonstrate overfitting and LASSO
# Generate data
N = 20
x = range(-1, stop=1,length=N)
a = [1, 0.5, 0.5, 1.5, 1]
y = a[1]*Pl.(x,0) + a[2]*Pl.(x,1) + a[3]*Pl.(x,2) +
    a[4]*Pl.(x,3) + a[5]*Pl.(x,4) + 0.25*randn(N)

# Solve LASSO using Convex.jl
d = 20
X = zeros(N,d)
for p = 0:d-1
    X[:,p+1] = Pl.(x,p)
end

lambda = 2
theta = LassoEN(y,X,lambda)

# Plot results
t    = range(-1, stop=1, length=200)
Xhat = zeros(length(t),d)
for p = 0:d-1
    Xhat[:,p+1] = Pl.(t,p)
end
yhat = Xhat*theta

p11 = scatter(x,y,makershape=:circle,label="data",legend=:bottomleft)
plot!(t,yhat,linecolor=:blue,linewidth=4,label="fitted curve")
display(p11)
#------------------------------------------------------------------------------
