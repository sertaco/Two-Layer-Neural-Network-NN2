function [Theta1, Theta2] =  NN2(X,y,hidden_layer_size,MaxIter,lambda)
% NN2 implements a two-layer neural network with variable input, hidden and
% output layer sizes. NN2 trains this NN for the given sample matrix
% X and output matrix y. NN2 can be used for multiclass classification.
% The number of layers is fixed to three.

% X is the training set with nxm dimensions, where m is
% the number of training samples and n is the number of features in each
% sample. y is a rxm matrix where r is the number of output classes. 

%Needed m files:
% -randInitializeWeights.m
% -nn2CostFunction.m
% -sigmoidGradient.m
% -fmincg.m
% -hout.m
% -sigmoid.m
% -checkNNGradients.m (optional)

%% Some Useful Variables:
m = size(X,2); % number of training samples
n = size(X,1); % number of features (input_layer_size)
r = size(y,1); % number of classes (num_labels or output_layer_size)
h = hidden_layer_size; % number of hidden layer units (hidden_layer_size)

%% Initializing Pameters
% In this part we implment a two layer neural network. First we initialize 
% the weights of the neural network (randInitializeWeights.m needed)

initial_Theta1 = randInitializeWeights(n, h);
initial_Theta2 = randInitializeWeights(h, r);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%% Check gradients by running checkNNGradients (Optional)
% checkNNGradients;

%% Training NN
% The cost function for two layer NN is implemented in nnCostFunction.m
% To train the NN, we will now use "fmincg". These
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', MaxIter);

costFunction = @(p) nn2CostFunction(p, n, h, r, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:h*(n+1)), h, (n+1));
Theta2 = reshape(nn_params((1+(h*(n+1))):end), r, (h+1));
end

