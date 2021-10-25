
%%    Shahab Sotudian 
% Regression  Assignment   Question 2.2
clc;
clear;
close all;

%% Ravesh sarih
% DATA 1
load('X1.mat')
load('Y1.mat')
Outlier_detection_linear_nonlinear_regression( X1,Y1 ,1)

% DATA 2
load('X2.mat')
load('Y2.mat')
Outlier_detection_linear_nonlinear_regression( X2,Y2 ,2)

% DATA 3
load('X3.mat')
load('Y3.mat')
Outlier_detection_linear_nonlinear_regression( X3,Y3 ,3)

% DATA 4
load('X4.mat')
load('Y4.mat')
Outlier_detection_linear_nonlinear_regression( X4,Y4 ,4)

% DATA 5
load('X5.mat')
load('Y5.mat')
Outlier_detection_linear_nonlinear_regression( X5,Y5 ,5)




%% Gradient_descent

% DATA 1
load('X1.mat')
load('Y1.mat')
Outlier_detection_Gradient_descent_regression( X1,Y1 ,1)

% DATA 2
load('X2.mat')
load('Y2.mat')
Outlier_detection_Gradient_descent_regression( X2,Y2 ,2)

% DATA 3
load('X3.mat')
load('Y3.mat')
Outlier_detection_Gradient_descent_regression( X3,Y3 ,3)

% DATA 4
load('X4.mat')
load('Y4.mat')
Outlier_detection_Gradient_descent_regression( X4,Y4 ,4)

% DATA 5
load('X5.mat')
load('Y5.mat')
Outlier_detection_Gradient_descent_regression( X5,Y5 ,5)














