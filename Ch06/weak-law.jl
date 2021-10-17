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
