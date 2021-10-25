% Shahab Sotudian - 94125091
% 7th Machine Learning Assignment  --  ANN
clear
clc
close all
%% Data
load('MNIST.mat')
n =size(MNIST, 1);                              % number of samples in the dataset
MNIST_targets  = MNIST(:,end);                  % last column is |label|
MNIST_targets(MNIST_targets == 0) = 10;         % use '10' to present '0'
MNIST_targetsd = dummyvar(MNIST_targets);       % convert label into a dummy variable
MNIST_inputs = MNIST(:,1:(end-1));              % the rest of columns are predictors
MNIST_inputs = MNIST_inputs';                   % transpose input
MNIST_targetsd = MNIST_targetsd';               % transpose dummy variable

x = MNIST_inputs;
t = MNIST_targetsd;


%% Create a Pattern Recognition Network

% Choose a Training Function
trainFcn = 'trainscg';                          % Scaled conjugate gradient backpropagation.uses less memory. Suitable in low memory situations.

% number of hidden layer neurons
hiddenLayerSize1=200;
hiddenLayerSize2=200;
net = patternnet([hiddenLayerSize1,hiddenLayerSize2]);

% Setup Division of Data for Training, Validation, Testing
% [trainInd,valInd,testInd] = dividerand(5999,0.70,0.15,0.15);
load('Data_division_indices.mat')
net.divideFcn='divideind'; 
net.divideParam.trainInd=trainInd;
net.divideParam.valInd=valInd;
net.divideParam.testInd=testInd;

net = configure(net,x,t);
rng(0)
net.IW{1,1}=0.001*ones(200,664);
net.b{1}=0.001*ones(200,1);
net.b{2}=0.001*ones(200,1);
net.b{3}=0.001*ones(10,1);

% Train the Network
[net,tr] = train(net,x,t);


%% PLOTTING

%  Data indices
I_test=tr.testInd;
I_train=tr.trainInd;
I_validation=tr.valInd;

% Test Confusion Matrix
Test_data=x(:,I_test);
y_test = net(Test_data);
t_test =t(:,I_test);
figure;
plotconfusion(t_test,y_test)
title('Test Confusion Matrix')
tind_test = vec2ind(t_test);
yind_test = vec2ind(y_test);
Test_Errors = 100*(sum(tind_test ~= yind_test)/numel(tind_test));
fprintf(['\n Classification Error for Test data =  ',' = %.3f%','%\n'], Test_Errors)


% Train Confusion Matrix
Train_data=x(:,I_train);
y_Train = net(Train_data);
t_Train =t(:,I_train);
figure;
plotconfusion(t_Train,y_Train)
title('Train Confusion Matrix')
tind_Train = vec2ind(t_Train);
yind_Train = vec2ind(y_Train);
Train_Errors = 100*(sum(tind_Train ~= yind_Train)/numel(tind_Train));
fprintf(['\n Classification Error for Train data =  ',' = %.3f%','%\n'], Train_Errors)


% Validation Confusion Matrix
validation_data=x(:,I_validation);
y_validation = net(validation_data);
t_validation =t(:,I_validation);
figure;
plotconfusion(t_validation,y_validation)
title('Validation Confusion Matrix')
tind_validation = vec2ind(t_validation);
yind_validation = vec2ind(y_validation);
validation_Errors = 100*(sum(tind_validation ~= yind_validation)/numel(tind_validation));
fprintf(['\n Classification Error for validation data =  ',' = %.3f%','%\n'], validation_Errors)


% All data Confusion Matrix
figure
y = net(x);
plotconfusion(t,y)
title('Confusion Matrix for all data')
tind = vec2ind(t);
yind = vec2ind(y);
all_Errors = 100*(sum(tind ~= yind)/numel(tind));
fprintf(['\n Classification Error for all data =  ',' = %.3f%','%\n'], all_Errors)

% performance plot
figure, plotperform(tr)


% View the Network
view(net)
