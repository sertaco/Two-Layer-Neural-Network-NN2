function [a3, z3, a2, z2,a1] = hout(X,Theta1,Theta2)
a1 = [ones(1,size(X,2));X];
z2 = Theta1*a1;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2));a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);
end
