function xbest = GPUCB(f, xinit, yinit, beta_t, sigma_n2, T, ppi, iter, debug)
% GPUCB - Applies the GP-UCB algorithm to f to determine its approximate maximum.
% 
% Syntax: xbest = GPUCB(f, xinit, yinit, beta_t, sigma_n2, T, ppi, stoppingCriterion, iter, debug)
%
% Inputs:
%       f - objective function R -> R
%       xinit - training points, must not be empty
%       yinit - values of f at xinit
%       beta_t - \beta_t = (sqrt \beta_t)^2 used in the algorithm.
%       sigma_n2 - value of sigma_n^2 in the noisy case
%       T - T = [T(1), T(2)], the search interval
%       ppi - amount of sampling points of calcUCB(x) within [n,n+1] in the search interval used to determine its maximum
%       iter - amount of iterations
%       debug - boolean, prints debug information and plot if set to true.
%
%   Outputs:
%       xbest - approximate position of the optimum of f.
%
%   Examples:
%       Example 1:
%           beta_t = @(t) 2*log(t^2*2*pi^2/(3*0.75))+2*log(t^2*7.5*sqrt(log(4/0.75)));
%           f=@(x) (x-2).*(x-5).*(x-7); xinit = 5.5; yinit = f(xinit);
%           xopt = GPUCB(f,xinit,yinit,beta_t,0,[0,8],100,20,false);
%       Example 2:
%           beta_t = @(t) 1;
%           f = @(x) -x.^2+4.3*x; xinit = linspace(0,4,3); yinit = f(xinit);
%           xopt = GPUCB(f,xinit,yinit,beta_t,0,[0,4],100,50,false);
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------


% Definitions
k = @(x,y,sigma_f2,l) sigma_f2*exp(-(((x-y)^2)/(2*l^2)));

[a,b] = size(xinit);
if (b > a) xinit = xinit'; end
[a,b] = size(yinit);
if (b > a) yinit = yinit'; end

x = xinit; % Vector of training values
y = yinit;
listLength = round(ppi*(T(2)-T(1)));

% training values
[ybest, index] = findMax(y);
xbest = x(index);

xlist = linspace(T(1),T(2),listLength);
UCBlist = zeros(listLength,1);

i = 1; % Iterator

for i = 1:iter
    % Find new hyperparameters
    [sigma_f2,l] = maximizeParams(x,y,k,sigma_n2);
    
    if (debug)
        fprintf('Maximization Parameters: %.5f, %.5f\n', sigma_f2, l);
    end
    
    % Find new evaluation point
    for j = 1:listLength
        UCBlist(j) = calcUCB(xlist(j),x,y,k,beta_t(j),sigma_f2,l,sigma_n2);
    end
    [~, index] = findMax(UCBlist);
    xnew = xlist(index);
    ynew = f(xnew);
    
    % Test if its bigger
    if (ynew >= ybest)
        ybest = ynew;
        xbest = xnew;
    end
    
    x = [x;xnew];
    y = [y;ynew];
    
    if (debug)
        fprintf('[%d] - Current Value: (%.5f, %.5f), beta_t: %.5f, Current UCB: %.5f, Current Max: (%.5f, %.5f)\n', i, xnew, ynew, beta_t(i), UCBlist(index), xbest, ybest);
    end
end

if (debug)
    fprintf('Stopping - (xmax,fmax) = (%.5f,%.5f)\n', xbest, ybest);
    fprintf('Calculating Plot...\n');

    % Plot Results at i %
    % objective function
    plotf_xlist = linspace(T(1),T(2),300*(T(2)-T(1)))';
    plotf_ylist = f(plotf_xlist);

    % Model - we draw last step's model.
    plotmodel_xlist = linspace(T(1),T(2),300*(T(2)-T(1)));
    plotmodel_ylist = zeros(length(plotmodel_xlist),1);
    x = x(1:length(x)-1);
    y = y(1:length(y)-1);
    for i = 1:length(plotmodel_xlist)
        plotmodel_ylist(i) = GP(plotmodel_xlist(i),x,y,k,sigma_f2,l,sigma_n2);
    end

    % UCB
    plotUCB_xlist = xlist;
    plotUCB_ylist = UCBlist;

    % Exact result
    plotexactresult_xlist = linspace(T(1),T(2),10000*(T(2)-T(1)));
    plotexactresult_ylist = f(plotexactresult_xlist);
    [ybestexact, index] = findMax(plotexactresult_ylist);
    xbestexact = plotexactresult_xlist(index);

    % Plot everything
    handle = zeros(6,1); % handles for the plots; used to exclude some plots from the legend
    handle(1) = plot(plotf_xlist,plotf_ylist,'b-','LineWidth',2);
    hold on;
    handle(2) = plot(plotmodel_xlist,plotmodel_ylist,'r-','LineWidth',2);
    plot(x,y,'rx','MarkerSize',8,'LineWidth',2);
    handle(3) = plot(xnew, ynew,'+','Color',[1 0.5 0],'MarkerSize',8,'LineWidth',2);
    handle(4) = plot(plotUCB_xlist,plotUCB_ylist,'g-','LineWidth',2);
    handle(5) = plot(xbest,ybest,'x','Color',[0 0.9 0.9],'MarkerSize',12,'LineWidth',2);
    handle(6) = plot(xbestexact,ybestexact,'k*','MarkerSize',8);
    xlabel('x');
    ylabel('f(x)');
    title('GP-UCB results','FontWeight','bold');
    legend(handle,'objective function','model','next test value','UCB','GP-UCB result','exact result','Location','Best');
    grid on;
    hold off;
    
    % Display the cumulative regret
    R_T = 0;
    for j = 1:iter
        R_T = R_T - y(j);
    end
    R_T = R_T + iter*ybestexact;
    fprintf('Cumulative Regret: %.5f\n', R_T);
end

end
