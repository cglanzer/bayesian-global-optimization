function xbest = EI(f, xinit, yinit, sigma_n2, T, ppi, stoppingCriterion, minIter, maxIter, debug)
% EI - Applies the expected improvement algorithm to f to determine its
% approximate maximum.
% 
% Syntax: xbest = EI(f, xinit, yinit, sigma_n2, T, ppi, stoppingCriterion, minIter, maxIter, debug)
%
% Inputs:
%       f - objective function R -> R
%       xinit - training points, must not be empty
%       yinit - values of f at xinit
%       sigma_n2 - value of sigma_n^2 in the noisy case
%       T - T = [T(1), T(2)], the search interval
%       ppi - amount of sampling points of EI(x) within [n,n+1] in the search interval used to determine its maximum
%       stoppingCriterion - the algorithm stops if max_x EI(x) < stoppingCriterion
%       minIter - minimum amount of iterations
%       maxIter - maximum amount of iterations
%       debug - boolean, prints debug information and plot if set to true.
%
%   Outputs:
%       xbest - approximate position of the optimum of f.
%
%   Examples:
%       Example 1:
%           f=@(x) (x-2).*(x-5).*(x-7); xinit = 5.5; yinit = f(xinit);
%           xopt = EI(f,xinit,yinit,0,[0,8],100,0.0001,5,1000,false);
%       Example 2:
%           f = @(x) -x.^2+4.3*x; xinit = linspace(0,4,3); yinit = f(xinit);
%           xopt = EI(f,xinit,yinit,0,[0,4],100,0.0001,5,1000,false);
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

% Training values
[ybest, index] = findMax(y);
xbest = x(index);

xlist = linspace(T(1),T(2),listLength);
EIlist = zeros(listLength,1);

i = 1; % Iterator

while (true)
    % Find new hyperparameters
    [sigma_f2,l] = maximizeParams(x,y,k,sigma_n2);
    
    if (debug)
        fprintf('Maximization Parameters: %.5f, %.5f\n', sigma_f2, l);
    end
    
    % Find new evaluation point
    % We calculate the maximum using brute force
    for j = 1:listLength
        EIlist(j) = calcEI(xlist(j),x,y,k,sigma_f2,l,sigma_n2);
    end
    % Index gives us the index of the maximum in xlist.
    [~, index] = findMax(EIlist);
    xnew = xlist(index);
    ynew = f(xnew);
    
    % Test if its bigger
    if (ynew >= ybest)
        ybest = ynew;
        xbest = xnew;
    end
    
    % Add new values to our training vectors
    x = [x;xnew];
    y = [y;ynew];
    
    if (debug)
        fprintf('[%d] - Current Value: (%.5f, %.5f), Current EI: %.5f, Current Max: (%.5f, %.5f)\n', i, xnew, ynew, EIlist(index), xbest, ybest);
    end
    
    % Stopping criterion
    if (i >= maxIter || (EIlist(index) < stoppingCriterion && i > minIter))
        if (debug)
            fprintf('Stopping - (xmax,fmax) = (%.5f,%.5f)\n', xbest, ybest);
            fprintf('Calculating Plot...\n');
            
            % Plot Results at i %
            % objective function
            plotf_xlist = linspace(T(1),T(2),300*(T(2)-T(1)))';
            plotf_ylist = f(plotf_xlist);
            
            % Model - go one step back (we plot the previous model)
            plotmodel_xlist = linspace(T(1),T(2),300*(T(2)-T(1)));
            plotmodel_ylist = zeros(length(plotmodel_xlist),1);
            x = x(1:length(x)-1);
            y = y(1:length(y)-1);
            for j = 1:length(plotmodel_xlist)
                plotmodel_ylist(j) = GP(plotmodel_xlist(j),x,y,k,sigma_f2,l,sigma_n2);
            end
            
            % EI
            plotEI_xlist = xlist;
            plotEI_ylist = EIlist;
            
            % Exact result, calculated using brute force
            plotexactresult_xlist = linspace(T(1),T(2),10000*(T(2)-T(1)));
            plotexactresult_ylist = f(plotexactresult_xlist);
            [ybestexact, index] = findMax(plotexactresult_ylist);
            xbestexact = plotexactresult_xlist(index);
            
            % Rescale factor
            % Find the optimum of EI to rescale it for the plotting process
            [xabsbestexact,~] = findMax(abs(plotexactresult_ylist));
            [eiabsmax,~] = findMax(abs(plotEI_ylist));
            rescalefactor = xabsbestexact/eiabsmax;
            plotEI_ylist = plotEI_ylist*rescalefactor;
            
            % Plot everything in a fancy way
            handle = zeros(6,1);
            handle(1) = plot(plotf_xlist,plotf_ylist,'b-','LineWidth',2);
            hold on;
            handle(2) = plot(plotmodel_xlist,plotmodel_ylist,'r-','LineWidth',2);
            plot(x,y,'rx','MarkerSize',9,'LineWidth',2);
            handle(3) = plot(xnew, ynew,'+','Color',[1 0.5 0],'MarkerSize',9,'LineWidth',2);
            handle(4) = plot(plotEI_xlist,plotEI_ylist,'g-','LineWidth',2);
            handle(5) = plot(xbestexact,ybestexact,'k*','MarkerSize',9);
            handle(6) = plot(xbest,ybest,'x','Color',[0 0.9 0.9],'MarkerSize',15,'LineWidth',2);
            xlabel('x');
            ylabel('f(x)');
            title('EI results','FontWeight','bold');
            legend(handle,'objective function','GP model','next test value','EI (rescaled)','exact result','EI result','Location','Best');
            grid on;
            hold off;
            
            % Display the cumulative regret
            R_T = 0;
            for j = 1:i
                R_T = R_T - y(j);
            end
            R_T = R_T + i*ybestexact;
            fprintf('Cumulative Regret: %.5f\n', R_T);
        end
        break;
    end
    
    i = i+1;
end
end
