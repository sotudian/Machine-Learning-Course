
%% Shahab sotudian  94125091
% Linear_Regression
function [ coefficient ] = Linear_Regression(X1,Y1)

n=size(X1,1);
x=X1;
y=Y1;
X=[x ones(n,1)];
Y=y;
coefficient=pinv(X)*Y;

end

