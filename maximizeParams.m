function [sigma_f2,l] = maximizeParams(trainingPoints,trainingResults,k,sigma_n2)
% MAXIMIZEPARAMS - maximizes the hyperparameters
% 
% Syntax: [sigma_f2, l] = maximizeParams(trainingPoints, trainingResults, k, sigma_n2)
%
% Inputs:
%       trainingPoints - training points, vector
%       trainingResults - values of f at trainingPoints
%       k - kernel function
%       sigma_n2 - value of sigma_n^2 referring to the expected noise
%
%   Outputs:
%       sigma_f2 - value of sigma_f^2 where the marginal likelihood is optimal
%       l - value of l where the marginal likelihood is optimal
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------

% We need column vectors
[a,b] = size(trainingPoints);
if (b > a) trainingPoints = trainingPoints'; end
[a,b] = size(trainingResults);
if (b > a) trainingResults = trainingResults'; end

% Maximize marginalLikelihood(x) using Matlab functions (fmincon).
% Note that fmindbnd does not work with multivariate problems like this
% one.
f=@(x) -marginalLikelihood(trainingPoints,trainingResults,k,x(1),x(2),sigma_n2);
options = optimset('Display', 'off');
[xres,~] = fmincon(f,[1,1],[],[],[],[],[1/(1e5),1/(1e5)],[1e5,1e5],[],options);
sigma_f2 = xres(1);
l = xres(2);
end