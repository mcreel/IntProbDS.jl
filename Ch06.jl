## -------------
##

# Julia code to compare the probability bounds
using Distributions, Plots

ϵ = 0.1
σ   = 1

p_exact(N) = 1 - cdf(Normal(), sqrt(N) * ϵ / σ)
p_cheby(N) = σ^2 / (ϵ^2 * N);
p_chern(N) = exp(-ϵ^2 * N / (2*σ^2));

ex_args    = (linewidth=2, color=RGB(1.0, 0.5, 0.0), label="Exact", markershape=:circle)
chb_args   = (linewidth=2, color=RGB(0.2, 0.7, 0.1), label="Chebyshev", linestyle=:dashdot)
chern_args = (linewidth=2, color=RGB(0.2, 0.0, 0.8), label="Chernoff")

Ns = [10^i for i in range(1, 3.8, length=50)]
scatter(Ns, p_exact.(Ns);  ex_args...,
        xaxis=:log10,
        yaxis=:log10, yticks=[1e-15, 1e-10, 1e-5, 1e0],
        legend=:bottomleft)
plot!(p_cheby; chb_args...)
plot!(p_chern; chern_args...)

##
## -------------
##

# Julia code to illustrate the weak law of large numbers
using Distributions, Plots

p = 0.5

# Nset = round.(Int, (10).^range(2,5,length=100))
# x = hcat((rand.(Binomial.(Nset, p), 1000)./Nset)...)
# y = x[1:10:end, :]'

Nset = round.((10).^range(2,5,length=100))
x = zeros(length(Nset), 1000)
for (i,N) in enumerate(Nset)
    x[i,:] = rand(Binomial(N, p), 1000) / N
end
y = x[:, 1:10:end]

scatter(Nset, y; xaxis=:log10, xticks=[1e2, 1e3, 1e4, 1e5],
        markershape=:x, color=:black,
        ylabel="sample average",
        legend=false)


plot!(Nset, p .+ 3*sqrt.(p*(1-p) ./ Nset); color=:red, linewidth=4)
plot!(Nset, p .- 3*sqrt.(p*(1-p) ./ Nset); color=:red, linewidth=4)

##
## -------------
##

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


##
## -------------
##

# Julia: Visualize convergence in distribution
using Distributions, Plots
N, p = 10, 0.5

B = Binomial(N,p)
Z = Normal(mean(B), std(B)) # N*p, sqrt(N*p*(1-p))

p_b(x) = pdf(B, x)
p_n(x) = pdf(Z, x)

c_b(x) = cdf(B, x)
c_n(x) = cdf(Z, x)

bin_args = (color = RGB(0, 0, 0), linewidth=2, label="Binomial")
norm_args = (color = RGB(0.8, 0, 0), linewidth=6, label="Normal")

ns = 0:N
p1 = plot(ns, p_b.(ns); bin_args..., seriestype=:sticks)
plot!(p_n, 0, N; norm_args...)

p2 = plot(c_b, 0, N; bin_args...)
plot!(c_n; norm_args...)

l = @layout [a b]
plot(p1, p2, layout=l)


##
## -------------
##

# Julia: Poisson to Gaussian: convergence in distribution
using Distributions, Plots

N = 4; # N = 10, N = 50;
λ = 1

P = Poisson(N*λ)
Z = Normal(mean(P), std(P))  # Nλ, √(Nλ)

p_b(x) = pdf(P, x) # evaluate on integers
p_n(x) = pdf(Z, x)

c_b(x) = cdf(P, x)
c_n(x) = cdf(Z, x)

bin_args = (linewidth=2, color=RGB(0.0, 0.0, 0.0), label="Poisson")
norm_args = (linewidth=6, colr=RGB(0.8, 0.0, 0.0), label="Gaussian")

ns = 0:2N
p1 = plot(ns, p_b.(ns); bin_args..., seriestype=:sticks, legend=:topright)
plot!(p_n, 0, 2N; norm_args...)

p2 = plot(c_b, 0, 2N; bin_args..., legend=:bottomright)
plot!(c_n; norm_args...)

l = @layout [a b]
plot(p1, p2, layout=l)

##
## -------------
##

# Julia: Visualize the Central Limit Theorem
using Distributions, Plots

N = 10
x = range(0, N, length=1001)
p = 0.5


B = Binomial(N,p)
Z = Normal(mean(B), std(B))  # N*p, sqrt(N*p*(1-p))

p_b(x) = pdf(B, x)
p_n(x) = pdf(Z, x)

c_b(x) = cdf(B, x)
c_n(x) = cdf(Z, x)


xs = range(5-2.5, 5 + 2.5, length=1001)

p1 = plot(p_n, 5-2.5, 5+2.5; fillrange=0 .* xs, color=RGB(0.6, 0.9, 1.0), legend=false,
     title="Binomial PDF")
plot!(0:N, p_b.(0:N); color=RGB(0,0,0), linewidth=2, seriestype=:sticks, layout=1)


p2 = plot(p_n, 5-2.5, 5+2.5, fillrange=0 .* xs, color=RGB(0.6, 0.9, 1.0), legend=false,
     title = "Gaussian PDF")
plot!(p_n, 0, N; color=RGB(0.8, 0, 0), linewidth=6, layout=2)

l = @layout [a b]
plot(p1, p2, layout=l)

##
## -------------
##

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

##
## -------------
##

# Julia: Failure of Central Limit Theorem at tails
using Distributions, Plots


λ = 1;

function gamma_pdf(N)
    function(x) # return anonymous function; also x -> ... notation
        (sqrt(N)/λ) * pdf(Gamma(N,λ), (x+sqrt(N))/(λ/sqrt(N)))
    end
end
gaussian_pdf(x) = pdf(Normal(), x)


plot(gamma_pdf(1), -1, 5; linewidth=4, color=RGB(0.8, 0.8, 0.8), label="N=1",
     yaxis=:log,
     legend=:bottomleft)
plot!(gamma_pdf(10);   linewidth=4, color=RGB(0.6, 0.6, 0.6), label="N=10")
plot!(gamma_pdf(100);  linewidth=4, color=RGB(0.4, 0.4, 0.4), label="N=100")
plot!(gamma_pdf(1000); linewidth=4, color=RGB(0.2, 0.2, 0.2), label="N=1000")
plot!(gaussian_pdf;    linewidth=4, color=RGB(0.9, 0.0, 0.0), label="Gaussian",
      linestyle=:dash)
