function [mu, sigma2] = GP(xstar, x, y, k, sigma_f2, l, sigma_n2)
% GP - calculates the GP posterior at xstar, given prior information.
% 
% Syntax: [mu, sigma2] = GP(xstar, x, y, k, sigma_f2, l, sigma_n2)
%
% Inputs:
%       xstar - vector of estimation points
%       x - training points
%       y - values of f at x
%       k - kernel function
%       sigma_f2 - value of sigma_f^2 to be used in k
%       l - value of l to be used in k
%       sigma_n2 - value of sigma_n^2 referring to the expected noise
%
%   Outputs:
%       mu - the expected posterior value \EE ( \xi(xstar) | (x,f(x)) )
%       sigma2 - the covariance matrix \Cov ( \xi(xstar) | (x,f(x)) )
%
%   Examples:
%       f = @(x) -x.^2+4.3*x;
%       k = @(x,y,sigma_f2,l) sigma_f2*exp(-(((x-y)^2)/(2*l^2)));
%       [mu, sigma2] = GP([1.5,3.5],[1,2,4,5],f([1,2,4,5]),k,1,1,0);
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------

% We need column vectors
[a,b] = size(xstar);
if (b > a) xstar = xstar'; end
[a,b] = size(x);
if (b > a) x = x'; end
[a,b] = size(y);
if (b > a) y = y'; end

% Calculate all the matrices needed
n = length(x);
m = length(xstar);
K = zeros(n,n);
Kstar = zeros(n,m);
Kstarstar = zeros(m,m);
for i = 1:n
    for j = 1:n
        K(i,j) = k(x(i), x(j), sigma_f2, l);
    end
end
for i = 1:n
    for j = 1:m
        Kstar(i,j) = k(x(i), xstar(j), sigma_f2, l);
    end
end
for i = 1:m
    for j = 1:m
        Kstarstar(i,j) = k(xstar(i), xstar(j), sigma_f2, l);
    end
end

% The following code avoids the problem of very small eigenvalues which are
% treated like zeros by Matlab. Otherwise, cholesky() doesn't work.
% Find out if there is an eigenvalue that is extremely small
SPD = true;
ew = eig(K);
for i = 1:length(ew)
    if (abs(ew(i)) < 1e-10) SPD = false; end
end
if (SPD == false) % If this is the case, increase those EWs a tiny bit
    K = K + 1e-5*eye(n);
end
% Note that the code above changes the result a bit. However, 1e-5
% is usually << stoppingCriterion in the EI algorithm, i.e. the influence is
% very small and can be ignored. However, this can lead to sigma2 being a
% very small negative value causing complex numbers to occur in the results
% (MATLAB(R) ignores them though). To avoid this, we return abs(sigma2).

R = chol(K+sigma_n2*eye(n));
mu = Kstar' * (R\(R'\y));
sigma2 = Kstarstar - Kstar' * (R\(R'\Kstar));
sigma2 = abs(sigma2); % See the note above
end