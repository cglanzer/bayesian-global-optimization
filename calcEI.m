function ret=calcEI(xstar, x, y, k, sigma_f2, l, sigma_n2)
% CALCEI - Calculates the expected improvement at xstar.
% 
% Syntax: result = calcEI(xstar, x, y, k, sigma_f2, l, sigma_n2)
%
% Inputs:
%       xstar - scalar value
%       x - training points
%       y - values of f at x
%       k - kernel function
%       sigma_f2 - value of sigma_f^2 to be used in k
%       l - value of l to be used in k
%       sigma_n2 - value of sigma_n^2 referring to the expected noise
%
%   Outputs:
%       ret - the expected improvement at xstar with training values x.
%
%   Examples:
%       f = @(x) -x.^2+4.3*x;
%       k = @(x,y,sigma_f2,l) sigma_f2*exp(-(((x-y)^2)/(2*l^2)));
%       ret=calcEI(3,[1,2,4,5],f([1,2,4,5]),k,1,1,0);
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------

% Calculate the GP posterior and M_n
[mu,sigma2] = GP(xstar, x, y, k, sigma_f2, l, sigma_n2);
[ybest,~] = findMax(y); % M_n

% This is the explicit formula by E. Vazquez and J. Bect (2010, eq. 11)
if (sigma2 > 0)
    u = (mu - ybest)/sqrt(sigma2);
    ret = sqrt(sigma2)*(normpdf(u) + u*normcdf(u));
elseif (mu > ybest)
    ret = mu-ybest;
else
    ret = 0;
end

end