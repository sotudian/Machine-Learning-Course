clc;
clear;
close all;

%%    Shahab Sotudian 
% Regression  Assignment  Question 2.1


% DATA 1
load('X1.mat')
load('Y1.mat')
Remove_Outliers( X1,Y1 ,1)

% DATA 2
load('X2.mat')
load('Y2.mat')
Remove_Outliers( X2,Y2 ,2)

% DATA 3
load('X3.mat')
load('Y3.mat')
Remove_Outliers( X3,Y3 ,3)

% DATA 4
load('X4.mat')
load('Y4.mat')
Remove_Outliers( X4,Y4 ,4)

% DATA 5
load('X5.mat')
load('Y5.mat')
Remove_Outliers( X5,Y5 ,5)



















