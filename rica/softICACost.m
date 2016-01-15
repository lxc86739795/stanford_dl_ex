%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)
% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%

% cost computation
cost = 0;

%numSamples = size(x,2);

cost = params.lambda.*sum(sum(sqrt((W*x).^2 + params.epsilon)));
cost = cost + 0.5.*sum(sum((W'*W*x - x).^2));

% gradient computation
Wgrad = W*(2.*(W'*W*x - x))*x' + 2.*(W*x)*(W'*W*x - x)'; 

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);