
clear
clc
load('MNIST.mat')

% %  Test---Train
% [trainInd,valInd,testInd] = dividerand(5999,0.75,0,0.25);
% MNIST_train=MNIST(trainInd',:);
% MNIST_test=MNIST(testInd',:);

% % Plotting
% figure                                          % plot images
% colormap(gray)                                  % set to grayscale
% for i = 1:25                                    % preview first 25 samples
%     subplot(5,5,i)                              % plot them in 6 x 6 grid
%     digit = reshape(MNIST(i, 1:(end-1)), [28,28])';    % row = 28 x 28 image
%     imagesc(digit)                              % show the image
%     title(num2str(MNIST(i,end)))                    % show the label
% end
 
% n =size(MNIST, 1);                    % number of samples in the dataset
% targets  = MNIST(:,end);                 % last column is |label|
% targets(targets == 0) = 10;         % use '10' to present '0'
% targetsd = dummyvar(targets);       % convert label into a dummy variable
% inputs = MNIST(:,1:(end-1));               % the rest of columns are predictors
% 
% inputs = inputs';                   % transpose input
% targets = targets';                 % transpose target
% targetsd = targetsd';               % transpose dummy variable
% 
% rng(1);                             % for reproducibility
% c = cvpartition(n,'Holdout',1500);   % hold out 1/3 of the dataset
% 
% Xtrain = inputs(:, training(c));    % 2/3 of the input for training
% Ytrain = targetsd(:, training(c));  % 2/3 of the target for training
% Xtest = inputs(:, test(c));         % 1/3 of the input for testing
% Ytest = targets(test(c));           % 1/3 of the target for testing
% Ytestd = targetsd(:, test(c));      % 1/3 of the dummy variable for testing

% create network
% net = network( ...
% 1, ... % numInputs, number of inputs,
% 2, ... % numLayers, number of layers
% [1; 0], ... % biasConnect, numLayers-by-1 Boolean vector,
% [1; 0], ... % inputConnect, numLayers-by-numInputs Boolean matrix,
% [0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix
% [0 1] ... % outputConnect, 1-by-numLayers Boolean vector
% );
% % View network structure
% % view(net);
% 
% 
% % number of hidden layer neurons
% net.layers{1}.size = 10;
% net.layers{2}.size = 20;
% % hidden layer transfer function
% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'logsig';
% % view(net);
% 
% % network training
% net.trainFcn = 'trainlm';
% net.performFcn = 'mse';
% net = train(net,inputs,outputs);
% % network response after training
% final_output = net(inputs)


n =size(MNIST, 1);                    % number of samples in the dataset
targets  = MNIST(:,end);                 % last column is |label|
targets(targets == 0) = 10;         % use '10' to present '0'
targetsd = dummyvar(targets);       % convert label into a dummy variable
inputs = MNIST(:,1:(end-1));               % the rest of columns are predictors

inputs = inputs(1:200,:)';                   % transpose input
% targets = targets';                 % transpose target
targetsd = targetsd(1:200,:)';               % transpose dummy variable

% % net = patternnet([3,4,5]);   
% x = inputs(:,1:100);
% t = targetsd(:,1:100);
% trainFcn = 'trainscg';           
% hiddenLayerSize = 100;                          
% net = patternnet(hiddenLayerSize);   
% 
% 
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% net.performFcn = 'crossentropy';
% 
% [net,tr] = train(net,x,t);
