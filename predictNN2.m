function [p,h2] = predictNN2(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% X must be a n X l matrix where n is the number of features, l is the
% number of samples to be predicted.

% Useful values
m = size(X, 2);

h1 = sigmoid(Theta1*[ones(1,m); X]);
h2 = sigmoid(Theta2*[ones(1,m); h1]);

p = h2>0.5;
end