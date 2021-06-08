function ret=calcUCB(xstar, x, y, k, beta_t, sigma_f2, l, sigma_n2)
% CALCUCB - Calculates the upper confidence bound for given beta_t.
% 
% Syntax: result = calcUCB(xstar, x, y, k, beta_t, sigma_f2, l, sigma_n2)
%
% Inputs:
%       xstar - scalar value
%       x - training points
%       y - values of f at x
%       k - kernel function
%       beta_t - \beta_t = (sqrt \beta_t)^2 used in the algorithm.
%       sigma_f2 - value of sigma_f^2 to be used in k
%       l - value of l to be used in k
%       sigma_n2 - value of sigma_n^2 referring to the expected noise
%
%   Outputs:
%       ret - the expected improvement at xstar with training values x.
%
%   Examples:
%       f = @(x) -x.^2+4.3*x;
%       beta_t = @(t) 2*log(t^2*2*pi^2/(3*0.75))+2*log(t^2*7.5*sqrt(log(4/0.75)));
%       k = @(x,y,sigma_f2,l) sigma_f2*exp(-(((x-y)^2)/(2*l^2)));
%       calcUCB(3,[1,2,4,5],f([1,2,4,5]),k,beta_t,1,1,0);
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------

[mu,sigma2] = GP(xstar, x, y, k, sigma_f2, l, sigma_n2);
ret = mu + sqrt(beta_t)*sqrt(sigma2);
end