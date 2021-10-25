function [ parameters ] = Gradient_Descent_linear( X1,Y1 )

x=X1;
y=Y1;

% Adding a column of ones to the beginning of the 'x' matrix
x = [ones(length(x), 1) x(:,1) x(:,2) x(:,3) x(:,4)];


% Running gradient descent on the data
parameters = [0;0;0;0;0];
learningRate = 0.01;
repetition = 1000;
m = length(y);   % Getting the length of our dataset
costHistory = zeros(repetition, 1);  % Creating a matrix of zeros for storing our cost function history
% Running gradient descent
for i = 1:repetition        
    % Calculating the transpose of our hypothesis
    h = (x * parameters - y)';        
    % Updating the parameters
    parameters(1) = parameters(1) - learningRate * (1/m) * h * x(:, 1);
    parameters(2) = parameters(2) - learningRate * (1/m) * h * x(:, 2);   
    parameters(3) = parameters(3) - learningRate * (1/m) * h * x(:, 3);  
    parameters(4) = parameters(4) - learningRate * (1/m) * h * x(:, 4);  
    parameters(5) = parameters(5) - learningRate * (1/m) * h * x(:, 5);  
    
    % Keeping track of the cost function
    
     

    %   Calculates the cost function

    costHistory(i) = (x * parameters - y)' * (x * parameters - y) / (2 * length(y));    
    
    
    
    
end


parameters=flip(parameters);



end

