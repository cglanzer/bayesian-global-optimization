% Bachelor Thesis, Christoph Glanzer
% Examples on how to use the scripts

% This script first defines the three example functions from the thesis.

% First function
f1=@(x) (x-2).*(x-5).*(x-7);
T1=[1,7.75];

% Second function
% (interpolating polynomial of
% {-6,10},{-5,5},{-4,-2},{-3,0},{-2,3},{-1,5},{0,7},{1,2},{2,3},{3,9},{4,7},{5,0})
f2=@(x) -(7*x.^11)/237600-(109*x.^10)/604800+(19*x.^9)/11340+(139*x.^8)/13440-(1213*x.^7)/37800-(5887*x.^6)/28800+(79*x.^5)/360+(200009*x.^4)/120960-(25267*x.^3)/453600-(249947 *x.^2)/50400-(9055*x)/5544+7;
T2=[-6,5];

% Third function
f3=@(x) sin(x.^3).*x/5;
T3=[-2.12,2.5];

% Feel free to adapt \delta or \beta_t here.
% Definitions
delta = 0.25;
betat = @(t) 2*log(t^2 * 2 * pi^2 / (3*delta)) + 2*log(t^2 * (T1(2)-T1(1)) * sqrt(log(4/delta)));

% Choose the training points here. As an example, we choose {2,6.75}.
% To choose a grid, use xinit=linspace(T1(1),T1(2),amount) where amount
% refers to the amount of points in the grid. Remember to change T1 if you
% want to simulate a function different from f_1.
% Training Points
xinit = [2,6.75];
yinit = f1(xinit);

% Uncomment the algorithm you want to apply
% For details on the arguments, please look at the descriptions in the
% files EI.m, resp. GPUCB.m
% Please update the arguments according to which function you want to use.
% The example uses f_1.
EI(f1,xinit,yinit,0,T1,100,0.001,30,30,true);
%GPUCB(f1,xinit,yinit,betat,0,T1,100,30,true);

% If needed, uncomment this to save the plot as .eps file.
%saveas(gcf, 'plot.eps', 'epsc');