function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
mean_value = mean(x,1);
x = x - repmat(mean_value, size(x, 1), 1);
sigma = x*x'/size(x, 2);
[U, S, V] = svd(sigma);
xRot = U'*x;
xPCAWhite = diag(1./sqrt(diag(S)+ epsilon))*xRot;
xZCAWhite = U*xPCAWhite;
Z = xZCAWhite;