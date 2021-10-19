clear all
close all
clc


y = load('ch10_LPC_data_02.txt');
K = 10;
N = 320;
Nblock = floor(length(y)/N);
M = Nblock*N;


% Determine Prediction Error
alpha = zeros(K,Nblock);
e = zeros(N,Nblock);
for i=1
    yblock = [y(N*(i-1)+[1:N])];
    y_corr = xcorr(yblock);
    YY     = toeplitz(y_corr(N+[0:K-1]));
    bb     = y_corr(N+[1:K]);
    h      = YY\bb;
end


figure;
y    = yblock;
z    = y(311:320);
yhat = zeros(340,1);
yhat(1:320) = y;
for t = 1:20
    predict     = z'*h;
    z           = [z(2:10); predict];
    yhat(320+t) = predict;
end
plot(yhat, 'r-', 'LineWidth', 4); hold on;
plot(y,    '-.', 'LineWidth', 4, 'color', [0.5,0.5,0.5]);
legend('Prediction','Input','Location','SW');
grid on;
set(gcf, 'Position', [100, 100, 600, 400]);
set(gca,'FontWeight','bold','fontsize',14);
% % export_fig ch10_LPC_03a.pdf -transparent -RGB;

