# Chapter 8.1 Maximum-likelihood Estimation

# Visualizing the likelihood function
## Julia: Visualize the likelihood function, surface plot
using StatsPlots
plotlyjs() # this backend allows rotating with mouse
N = 50
S = range(1., N, step=0.1)
θ = range(0.1, 0.9,length=100)
L(S, θ) = S*log(θ) + (N-S)*log(1. - θ)
surface(S, θ, (S, θ) -> L(S, θ), color=:jet)
xlabel!("S")
ylabel!("θ")
title!("ℒ(θ|S)") 

## Visualizing the likelihood function, heatmap
using StatsPlots
gr()
N = 50
S = range(1., N, step=0.1)
θ = range(0.1, 0.9,length=100)
L(S, θ) = S*log(θ) + (N-S)*log(1. - θ)
heatmap(S, θ, (S, θ) -> L(S, θ), color=:jet)
xlabel!("S")
ylabel!("θ")
title!("ℒ(θ|S)") 

## Visualizing the likelihood function, slice 
using StatsPlots
gr()
N = 50
S = 25  # set value of S here
θ = range(0.1, 0.9,length=100)
L(θ) = S*log(θ) + (N-S)*log(1. - θ)
plot(θ, θ -> L(θ),label=false)
xlabel!("θ")
title!("ℒ(θ|S=$S)") 


# ML estimation for single-photon imaging
# Julia code
using Images, Distributions
using ImageView:imshow
download("https://probability4datascience.com/data/cameraman.tif", "./cameraman.tif")
λ  = Float64.(load("cameraman.tif"))
T = 100
x = [rand.(Poisson.(λ)) for _=1:T]
y = [x[i] .>= 1. for i=1:T]
λhat = -log.(1. .- mean(y))
imshow(x[1])
imshow(λhat)


# Chapter 8.2 Properties of the ML estimation

# Visualizing the invariance principle
# Julia code
using StatsPlots
N = 50
S = 20
θ = range(0.1,0.9,length=1000)
η  = -log.(1. .- θ) # the reparameterization transformation
L₁(θ) = S*log(θ) + (N-S)*log(1. - θ) # log likelihood in terms of θ 
L₂(η) = S*log(1. - exp(-η)) - (N-S)*η # log likelihood in terms of η 
# plot the log likelihood with θ parameterization 
p4 = plot(θ, θ -> L₁(θ), linewidth=5, color=:blue, label=false)
xlabel!("θ")
title!("ℒ₁(θ|S=$S)") 
vline!([0.4], label=false, color=:red)
# plot the transformation
p2 = plot(θ, η, linewidth=5, color=:black, label=false)
xlabel!("θ")
ylabel!("η") 
title!("η(θ)") 
vline!([0.4], label=false, color=:red)
hline!([0.5], label=false, color=:green)
# plot the log likelihood with η parameterization
p1 = plot(L₂.(η), η,  linewidth=5, color=:blue, label=false, xflip=true)
ylabel!("η") 
hline!([0.5], label=false, color=:green)
title!("ℒ₂(η|S=$S)")
# blank place holder plot
p3 = plot(grid=false, xaxis=false, yaxis=false, xticks=false, yticks=false)
# plot the whole set in the final image
plot(p1, p2, p3, p4)


#=
# Chapter 8.3 Maximum-a-Posteriori Estimation
# Influence of the priors
# Julia code
N = 1;
sigma0 = 1;
mu0    = 0.0;
x = 5;%*rand(N,1);
t = linspace(-3,7,1000);

q = NaN(1,1000);
for i=1:N
    [val,idx] = min(abs(t-x(i)));
    q(idx) = 0.1;
end

p0 = normpdf(t,0,1);
theta = (mean(x)*sigma0^2+mu0/N)/(sigma0^2+1/N);
p1 = normpdf(t,theta,1);
prior = normpdf(t,mu0,sigma0)/10;

figure;
h1 = plot(t,normpdf(t,mean(x),1),'LineWidth',4,'Color',[0.8,0.8,0.8]); hold on;
h2 = plot(t,p1,'LineWidth',4,'Color',[0,0.0,0.0]);
h3 = stem(t,q,'LineWidth',4,'Color',[0.5,0.5,0.5],'MarkerSize',10);
h5 = plot(t,prior,'LineWidth',4,'Color',[1,0.5,0.0]);

grid on;
axis([-3 7 0 0.5]);
xticks(-5:1:10);
yticks(0:0.05:0.5);
set(gcf, 'Position', [100, 100, 600, 300]);
set(gca,'FontWeight','bold','fontsize',14);

# Conjugate priors
# Julia code
sigma0 = 0.25;
mu0    = 0.0;

mu     = 1;
sigma  = 0.25;

Nset = [0 1 2 5 8 12 20];
x0 = sigma*randn(100,1);

for i=1:7
    N = Nset(i);
    x = x0(1:N);
    t = linspace(-1,1.5,1000);

    p0     = normpdf(t,0,1);
    theta  = mu*(N*sigma0^2)/(N*sigma0^2+sigma^2) + mu0*(sigma^2)/(N*sigma0^2+sigma^2);
    sigmaN = sqrt(1/(1/sigma0^2+N/sigma^2));
    posterior(:,i) = normpdf(t,theta,sigmaN);
end

figure;
plot(t,posterior(:,1)', 'LineWidth',2,'Color',[0.9,0.0,0.0]);  hold on;
plot(t,posterior(:,2)', 'LineWidth',2,'Color',[1,0.6,0.0]);
plot(t,posterior(:,3)', 'LineWidth',2,'Color',[1,0.9,0.3]);
plot(t,posterior(:,4)', 'LineWidth',2,'Color',[0.0,0.8,0.1]);
plot(t,posterior(:,5)', 'LineWidth',2,'Color',[0.0,0.6,0.6]);
plot(t,posterior(:,6)', 'LineWidth',2,'Color',[0.0,0.2,0.8]);
plot(t,posterior(:,7)', 'LineWidth',2,'Color',[0.5,0.2,0.5]);

legend('N = 0', 'N = 1', 'N = 2', 'N = 5', 'N = 8', 'N = 12', 'N = 20', 'Location', 'NW');
grid on;
axis([-1 1.5 0 8]);
set(gcf, 'Position', [100, 100, 600, 300]);
set(gca,'FontWeight','bold','fontsize',14);

=#
