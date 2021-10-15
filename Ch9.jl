# Julia code for Chapter 9

# Chapter 9.1 Confidence Interval

####################################################################
## Histogram of the sample average
using Distributions, Plots, Printf
gmm = MixtureModel(Normal[Normal(0, 1), Normal(5, 1),], [0.3, 0.7])
x = range(-5., 10., length=1000)
# the population distribution
plot(x, pdf(gmm, x), linewidth=3, label=false, title="Population distribution")
# histograms of 4 random samples of size 50
plot_array = [] 
for i=1:4
    bin = -5:10
    Y = rand(gmm, 50)
    tt = @sprintf("Mean = %3.2f", mean(Y))
    push!(plot_array, histogram(Y, bins=bin, label = tt, legend=:topleft))
end
plot(plot_array...)
bin = 2:0.1:5;
for i=1:10000
    display()
    Y(:,i) = random(gmm, 50);
end
# histogram of sample average
M = zeros(10000)
bin = range(2., 5., step=0.1)
for i = 1:10000
    M[i] = mean(rand(gmm, 50))
end   
histogram(M,bins=bin, label=false, title="Histogram of the sample average")

####################################################################
## Compute confidence interval
# Julia code to compute the width of the confidence interval
using Distributions
alpha = 0.05
mu = 0.
sigma = 1.
epsilon = quantile(Normal(mu, sigma),1. - alpha/2.) # CI is estimator \

####################################################################
## Visualize the t-distribution
# Julia code to plot the t-distribution
x = linspace(-5,5,100);
p1 = pdf('norm',x,0,1);
p2 = pdf('t',x,11-1);
p3 = pdf('t',x,3-1);
p4 = pdf('t',x,2-1);

figure;
plot(x,p1,'-.','LineWidth',4,'Color',[0 0.6 0]); hold on;
plot(x,p2,'-o','LineWidth',2,'Color',[1 0.6 0.2]);
plot(x,p3,'-^','LineWidth',2,'Color',[0.2 0.2 0.7]);
plot(x,p4,'-s','LineWidth',2,'Color',[0.7 0.2 0.2]);
legend('Gaussian(0,1)', 't-dist, N = 11', 't-dist, N = 3', 't-dist, N = 2');
grid on;
xticks(-5:1:5);
yticks(0:0.05:0.4);
set(gcf, 'Position', [100, 100, 600, 300]);
set(gca,'FontWeight','bold','fontsize',14);

####################################################################
## Construct a confidence interval from data
# Julia code to generate a confidence interval
import numpy as np
import scipy.stats as stats
x = np.array([72, 69, 75, 58, 67, 70, 60, 71, 59, 65])
Theta_hat = np.mean(x) # Sample mean
S_hat     = np.std(x)  # Sample standard deviation
nu        = x.size-1   # degrees of freedom
alpha     = 0.05       # confidence level
z    = stats.t.ppf(1-alph/2, nu)
CI_L = Theta_hat-z*S_hat/np.sqrt(N)
CI_U = Theta_hat+z*S_hat/np.sqrt(N)
print(CI_L, CI_U)

####################################################################

# Chapter 9.2 Bootstrap

####################################################################
## Visualize bootstrap
# Julia code to illustrate bootstrap
gmm = gmdistribution([0; 5], cat(3,1,1), [0.3 0.7]);
x = linspace(-5, 10, 1000)';
f = pdf(gmm, x);
figure;
plot(x,f,'LineWidth', 6,'color',[0.1, 0.4, 0.6]);
grid on;
axis([-5 10 0 0.35]);
legend('Population', 'Location', 'NW');
set(gcf, 'Position', [100, 100, 600, 300]);
set(gca,'FontWeight','bold','fontsize',14);

% Visualize bootstrap dataset
N = 100;
X = random(gmm, N);
figure;
[num,val] = hist(X, linspace(-5,10,30));
bar(val, num,'FaceColor',[0.1, 0.4, 0.6]);
set(gcf, 'Position', [100, 100, 600, 300]);
set(gca,'FontWeight','bold','fontsize',14);

for i=1:3
    % Bootstrap dataset
    idx = randi(N,[1, N]);
    Xb1 = X(idx);
    figure;
    [num,val] = hist(Xb1, linspace(-5,10,30));
    bar(val, num,'FaceColor',[0.1, 0.4, 0.6]);
    set(gcf, 'Position', [100, 100, 600, 300]);
    set(gca,'FontWeight','bold','fontsize',14);
end

% Bootstrap
K = 1000;
Thetahat = zeros(1,K);
for i=1:K                       % repeat K times
    idx = randi(N,[1, N]);      % sampling w/ replacement
    Y = X(idx);
    Thetahat(i) = median(Y);    % estimator
end
M = mean(Thetahat)              % bootstrap mean
V = var(Thetahat)               % bootstrap variance
figure;
[num,val] = hist(Thetahat, linspace(3.5,5.5,30));
bar(val, num,'FaceColor',[0.2, 0.2, 0.2]);
set(gcf, 'Position', [100, 100, 600, 300]);
set(gca,'FontWeight','bold','fontsize',14);

####################################################################
## Bootstrap median
# Julia code to estimate a bootstrap variance
X = [72, 69, 75, 58, 67, 70, 60, 71, 59, 65];
N = size(X,2);
K = 1000;
Thetahat = zeros(1,K);
for i=1:K                       % repeat K times
    idx = randi(N,[1, N]);      % sampling w/ replacement
    Y = X(idx);
    Thetahat(i) = median(Y);    % estimator
end
M = mean(Thetahat)              % bootstrap mean
V = var(Thetahat)               % bootstrap variance

####################################################################

# Chapter 9.3 Hypothesis Testing

####################################################################
## Estimate Z-value
% MATLAB command to estimate the Z_hat value.
Theta_hat = 0.29;                    % Your estimate
theta    = 0.25;                     % Your hypothesis
sigma    = sqrt(theta*(1-theta));    % Known standard deviation
N        = 1000;                     % Number of samples
Z_hat    = (Theta_hat - theta)/(sigma/sqrt(N));

####################################################################
## Compute critical value
% MATLAB code to compute the critical value
alpha = 0.05;
z_alpha = icdf('norm', 1-alpha, 0, 1);

####################################################################
##Compute p-value
% MATLAB code to compute the p-value
p = cdf('norm', -1.92, 0, 1);

####################################################################

# Chapter 9.5 ROC Curve

####################################################################
## Plot an ROC curve
# Julia code to plot ROC curve
sigma = 2;  mu = 3;
alphaset = linspace(0,1,1000);
PF1 = zeros(1,1000); PD1 = zeros(1,1000);
PF2 = zeros(1,1000); PD2 = zeros(1,1000);
for i=1:1000
    alpha = alphaset(i);
    PF1(i) = alpha;
    PD1(i) = alpha;

    PF2(i) = alpha;
    PD2(i) = 1-normcdf(norminv(1-alpha)-mu/sigma);
end
figure;
plot(PF1, PD1,'LineWidth', 4, 'Color', [0.8, 0, 0]); hold on;
plot(PF2, PD2,'LineWidth', 4, 'Color', [0, 0, 0]); hold off;

Computer area under curve
% MATLAB
auc1 = sum(PD1.*[0 diff(PF1)])
auc2 = sum(PD2.*[0 diff(PF2)])

####################################################################
## Another ROC curve
# Julia code to generate the ROC curve.
sigma = 2;  mu = 3;
PF1 = zeros(1,1000); PD1 = zeros(1,1000);
PF2 = zeros(1,1000); PD2 = zeros(1,1000);
alphaset = linspace(0,0.5,1000);
for i=1:1000
    alpha = alphaset(i);
    PF1(i) = 2*alpha;
    PD1(i) = 1-(normcdf(norminv(1-alpha)-mu/sigma)-...
                normcdf(-norminv(1-alpha)-mu/sigma));
end
alphaset = linspace(0,1,1000);
for i=1:1000
    alpha = alphaset(i);
    PF2(i) = alpha;
    PD2(i) = 1-normcdf(norminv(1-alpha)-mu/sigma);
end
figure;
plot(PF1, PD1,'LineWidth', 4, 'Color', [0.8, 0, 0]); hold on;
plot(PF2, PD2,'LineWidth', 4, 'Color', [0, 0, 0]); hold off;

####################################################################
## ROC on real data
 ch9_ROC_example_data.txt (790KB)
% MATLAB: construct data
% Do not worry if you cannot understand this code.
% It is not the focus on this book.
load fisheriris
pred = meas(51:end,1:2);
resp = (1:100)'>50;
mdl = fitglm(pred,resp,'Distribution','binomial','Link','logit');
scores = mdl.Fitted.Probability;
labels = [ones(1,50), zeros(1,50)];
save('ch9_ROC_example_data','scores','labels');