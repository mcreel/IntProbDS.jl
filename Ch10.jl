# Functions:
function fft_len(x::AbstractVector, len::Int) 
    if len > length(x)
        return fft([x; zeros(len - length(x))])
    else
        return fft(x[1:len])
    end
end

function conv_same(x::AbstractVector, y::AbstractVector)
    s = length(y) ÷ 2
    e = length(x) + s - 1
    return conv(x, y)[s:e]
end

####################################################################

# Julia code for Example 10.5
using Plots

x = zeros(1000, 20)
t = range(-2, 2, length=1000)

for col in eachcol(x)
    col .= rand() * cos.(2π * t)
end

plot(t, x, linewidth=2, color=:gray80, legend=false)
plot!(t, 0.5 * cos.(2π * t), linewidth=4, color=:darkred)

####################################################################

# Julia code for Example 10.6
using Plots

x = zeros(1000, 20)
t = range(-2, 2, length=1000)

for col in eachcol(x)
    col .= cos.(2π*t .+ 2π*rand())
end

plot(t, x, linewidth=2, color=:gray80, legend=false)
hline!([0], linewidth=4, color=:darkred)

####################################################################

# Julia code for Example 10.7
using Plots

x = zeros(21, 20)
t = 0:20

for col in eachcol(x)
    col .= rand() .^ t
end

sticks(t, x, linewidth=2, shape=:circle, color=:gray80, legend=false)
sticks!(t, 1 ./ (t .+ 1), linewidth=2, shape=:circle, markersize=6, color=:darkred)

####################################################################

# Julia code for Example 10.11: Plot the time function
using Plots

x = zeros(1000, 20)
t = range(-2, 2, length=1000)

for col in eachcol(x)
    col .= rand() * cos.(2π * t)
end    

plot(t, x, linewidth=2, color=:gray80, legend=false)
plot!(t, 0.5 * cos.(2π * t), linewidth=4, color=:darkred)
scatter!(zeros(20), x[501, :], markersize=4, color=:orange)
scatter!(0.5 * ones(20), x[626, :], markersize=4, color=:blue)

# Julia code for Example 10.11: Plot the autocorrelation function
t = range(-1, 1, length=1000)
R = 1/3 * cos.(2π * t) * cos.(2π * t')
 
heatmap(t, t, R, yticks= -1:0.25:1, xticks= -1:0.25:1, legend=false, yflip=true)

####################################################################

# Julia code for Example 10.12: Plot the time function
using ToeplitzMatrices
using Plots

x = zeros(1000, 20)
t = range(-2, 2, length=1000)

for col in eachcol(x)
    col .= cos.(2π*t .+ 2π*rand())
end

plot(t, x, linewidth=2, color=:gray80, legend=false)
plot!(t, 0 * cos.(2π * t), linewidth=4, color=:darkred)
scatter!(zeros(20), x[501, :], markersize=4, color=:orange)
scatter!(0.5 * ones(20), x[626, :], markersize=4, color=:blue)

# Julia code for Example 10.12: Plot the autocorrelation function
t = range(-1, 1, length=1000) 
R = SymmetricToeplitz(0.5 * cos.(2π * t))
 
heatmap(t, t, R, yticks= -1:0.25:1, xticks= -1:0.25:1, legend=false, yflip=true)

####################################################################

# Julia code to demonstrate autocorrelation
using Plots
using DSP

# Figure 1
X₁ = randn(1000)
X₂ = randn(1000)

plot(X₁, linewidth=2, linestyle=:dot, color=:blue, legend=false)
plot!(X₂, linewidth=2, linestyle=:dot, color=:black)

# Figure 2
N  = 1000 # number of sample paths
T  = 1000 # number of time stamps
X  = randn(N, T)
xc = zeros(N, 2T - 1)

for i in 1:N
    xc[i, :] .= xcorr(X[i, :], X[i, :]) / T
end

plot(xc[1, :], linewidth=2, linestyle=:dot, color=:blue, label="correlation of sample 1")
plot!(xc[2, :], linewidth=2, linestyle=:dot, color=:black, label="correlation of sample 2")
####################################################################

# Julia code for Example 10.15
using Plots, LaTeXStrings
using Images
using DSP

# Figure 1
t = -10:0.001:10
L = length(t)
X = randn(L)
h = 10 * max.(1 .- abs.(t), 0) / 1000
Y = imfilter(X, h, "circular")

plot(t, X, linewidth=0.5, color=:gray60, label=L"X(t)")
hline!([0], linewidth=4, color=:gray20, label=L"\mu_X(t)")
plot!(t, Y, linewidth=2, color=:red, label=L"Y(t)")
hline!([0], linewidth=4, linestyle=:dot, color=:yellow, label=L"\mu_Y(t)")
xlims!((-10, 10))

# Figure 2
h₂ = conv(h, h)
Rₓ = zeros(40001)
Rₓ[20001] = 0.2

plot(-20:0.001:20, Rₓ, linewidth=2, color=:gray60, label=L"R_X(t)")
plot!(-20:0.001:20, h₂, linewidth=2, color=:red, label=L"R_Y(t)")
xlims!((-2, 2))
ylims!((-0.05, 0.2))

####################################################################

# Julia code to solve the Yule Walker Equation
using Plots, LaTeXStrings
using ToeplitzMatrices
using DelimitedFiles
using DSP

y = vec(readdlm("ch10_LPC_data.txt"))
K = 10
N = length(y)
y_corr = xcorr(y, y)
R = SymmetricToeplitz(y_corr[N:N+K-1])
lhs = y_corr[N+1:N+K]
h = R \ lhs

# Figure 1
plot(y, linewidth=4, color=:blue, label=L"Y[n]")

# Figure 2
plot(y_corr, linewidth=4, color=:black, label=L"R_Y[k]")

####################################################################

# Julia code to predict the samples
using ToeplitzMatrices
using DelimitedFiles
using Plots
using DSP

y = vec(readdlm("ch10_LPC_data_02.txt"))
K = 10
N = length(y)

y_corr = xcorr(y, y)
R = SymmetricToeplitz(y_corr[N:N+K-1])
lhs = y_corr[N+1:N+K]
h = R \ lhs

z = y[311:320]
ŷ = zeros(340)
ŷ[1:320] .= y

for t in 1:20
    pred = z' * h
    z = [z[2:10]; pred]
    ŷ[320+t] = pred
end

plot(ŷ, linewidth=3, color=:red, label="Prediction", legend=:bottomleft)
plot!(y, linewidth=4, linestyle=:dash, color=:gray60, label="Input")

####################################################################

# Julia code for Wiener filtering
using DSP, FFTW
using Plots

w = 0.05 * randn(320)
x = y + w

Ry = xcorr(y, y)
Rw = xcorr(w, w)
Sy = fft(Ry)
Sw = fft(Rw)
H = Sy ./ (Sy + Sw)

Ŷ = H .* fft_len(x, 639)
ŷ = real.(ifft(Ŷ))

# Figure 1
plot(x, linewidth=4, color=:gray, label="Noise Input X[n]", legend=:bottomleft)
plot!(ŷ[1:320], linewidth=2, color=:red, label="Wiener Filtered Yhat[n]")
plot!(y, linewidth=2, linestyle=:dot, color=:black, label="Ground Truth Y[n]")

# Figure 2
plot(Rw, linewidth=4, color=:blue, label="h[n]")
####################################################################

# Julia code to solve the Wiener deconvolution problem
using DelimitedFiles
using DSP, FFTW
using Plots

y = vec(readdlm("ch10_wiener_deblur_data.txt"))
g = ones(32) / 32
w = 0.02 * randn(320)
x = conv_same(y, g) + w

Ry = xcorr(y, y)
Rw = xcorr(w, w)
Sy = fft(Ry)
Sw = fft(Rw)
G  = fft_len(g, 639)

H = @. (conj(G) * Sy) / (abs(G)^2 * Sy + Sw)
Ŷ = H .* fft_len(x, 639)
ŷ = real.(ifft(Ŷ))

plot(x, linewidth=4, color=:gray, label="Noise Input X[n]")
plot!(16:320+15, ŷ[1:320], linewidth=2, color=:red, label="Wiener Filtered Yhat[n]")
plot!(y, linewidth=2, linestyle=:dot, color=:black, label="Ground Truth Y[n]")
