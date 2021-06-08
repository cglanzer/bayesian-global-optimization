function ret=marginalLikelihood(x,y,k,sigma_f2,l,sigma_n2)
% MARGINALLIKELIHOOD - evaluates the marginal likelihood
% 
% Syntax: ret = marginalLikelihood(x, y, k, sigma_f2, l, sigma_n2)
%
% Inputs:
%       x - training points
%       y - values of f at x
%       k - kernel function
%       sigma_f2 - value of sigma_f^2 to be used in k
%       l - value of l to be used in k
%       sigma_n2 - value of sigma_n^2 referring to the expected noise
%
%   Outputs:
%       ret - the marginal likelihood evaluated using k
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------


% Calculate K
n = length(x);
K = zeros(n,n);
for i = 1:n
    for j = 1:n
        K(i,j) = k(x(i),x(j),sigma_f2,l);
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
% is usually << stoppingCriterion in the algorithm, i.e. the influence is
% very small and can be ignored.

% Calculate the marginal Likelihood
R = chol(K+sigma_n2*eye(n));
factor_sum = 0;
for i = 1:n
    factor_sum = factor_sum + log(R(i,i));
end
ret = (-1/2)*y'*(R\(R'\y))-factor_sum;
end
