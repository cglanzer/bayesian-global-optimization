function [ymax,index] = findMax(y)
% FINDMAX - Finds and returns the maximum of a vector.
% 
% Syntax: [ymax,index] = findMax(y)
%
% Inputs:
%       y - vector
%
%   Outputs:
%       index - the index of y such that ymax = y(index)
%       ymax - the biggest value within y
%
%   Examples:
%       y = [1;5;3;7;1];
%       [ymax,index] = findMax(y);
%       % Result: [7,4]
%
%   Author: Christoph Glanzer
%

%------------- BEGIN CODE --------------

ymax = y(1);
index = 1;
for i = 2:length(y)
    if (y(i) > ymax)
        ymax = y(i);
        index = i;
    end
end
end