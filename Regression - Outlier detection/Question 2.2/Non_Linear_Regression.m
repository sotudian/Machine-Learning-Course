
%% Shahab sotudian  94125091
% Linear_Regression
function [ coefficient ] = Non_Linear_Regression(X1,Y1)

n=size(X1,1);
x=X1;
X=[x.^2 x ones(n,1)];
Y=Y1;
coefficient=pinv(X)*Y;

end

