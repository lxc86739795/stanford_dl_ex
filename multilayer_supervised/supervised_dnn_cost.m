function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%

% compute activations of hidden layers
for l = 1:numHidden
    if(l == 1)
        z = bsxfun(@plus, stack{l}.W*data, stack{l}.b);
    else
        z = bsxfun(@plus, stack{l}.W*hAct{l-1}, stack{l}.b);
    end
     hAct{l} = sigmoid(z);
end

% compute softmax values of the output layer
h = bsxfun(@plus, (stack{numHidden + 1}.W)*hAct{numHidden}, stack{numHidden+1}.b);
e = exp(h);
pred_prob = bsxfun(@rdivide,e,sum(e,1));
hAct{numHidden+1} = pred_prob;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

% compute cross entropy cost of softmax
ceCost =0;
c = log(pred_prob);
I = sub2ind(size(c), labels', 1:size(c,2));%找出矩阵c的线性索引，行由labels指定，列由1:size(c,2)指定，生成线性索引返回给I
values = c(I);
ceCost = - sum(values);

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
y = zeros(size(pred_prob));
y(I) = 1;
error = pred_prob - y;

for l = numHidden + 1:-1:1
   gradStack{l}.b = sum(error, 2);
   if (l == 1)
       gradStack{l}.W = error*data';
       break;
   else
       gradStack{l}.W = error*hAct{l-1}';
   end
   error = (stack{l}.W)'*error.*hAct{l-1}.*(1-hAct{l-1});
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;

for l = 1:numHidden+1
  wCost = wCost + .5*ei.lambda*sum(stack{l}.W(:).^2);%所有权值的平方和
end

cost = ceCost + wCost;

% Computing the gradient of the weight decay.
for l = numHidden:-1:1
  gradStack{l}.W = gradStack{l}.W + ei.lambda*stack{l}.W;%softmax没用到权重衰减项
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



