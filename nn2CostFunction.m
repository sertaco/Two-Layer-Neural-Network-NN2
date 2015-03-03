function [J, grad] = nn2CostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m= size(X, 2);
% Forward propagation:         
[a3, ~, a2, z2,a1] = hout(X,Theta1,Theta2);
J = 1/m*(-log(a3).*y-log(1-a3).*(ones(num_labels,m)-y));
J=sum(sum(J));

% Regularization term:
Theta1_=Theta1(:,2:end);
Theta2_=Theta2(:,2:end);
J=J+lambda/2/m*(sum(sum(Theta1_.^2))+sum(sum(Theta2_.^2)));


% Backpropagation
del3 = a3 - y;
del2 = (Theta2_'*del3).*sigmoidGradient(z2);

Theta2_grad = 1/m.*(del3*a2');
Theta1_grad = 1/m.*(del2*a1');

% Regularization term
Theta2_grad = Theta2_grad + lambda/m*[zeros(size(Theta2_grad,1),1),Theta2_];
Theta1_grad = Theta1_grad + lambda/m*[zeros(size(Theta1_grad,1),1),Theta1_];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
