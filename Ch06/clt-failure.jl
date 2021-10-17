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
