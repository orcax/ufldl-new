function [ cost, grad, pred_prob] = supervised_dnn_cost(theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

M = size(data, 2); % number of training data

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
a1 = data; % 784*60000
% hidden layer
z2 = stack{1}.W * a1 + repmat(stack{1}.b, 1, M);
a2 = sigmoid(z2); % 256*M

% output layer
z3 = stack{2}.W * a2 + repmat(stack{2}.b, 1, M); % 10*M
a3 = sigmoid(z3); % 10*M
pred_prob = a3;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cross entropy cost by softmax
y = full(sparse(labels, 1:M, 1));
p = exp(z3);
p = bsxfun(@rdivide, p, sum(p));
cost = -1 / M * y(:)' * log(p(:));

%% compute gradients using backpropagation
% hidden layer
delta = - (y - p);
gradStack{2}.W = delta * a2' / M;
gradStack{2}.b = sum(delta, 2) / M;
% input layer
delta = (stack{2}.W' * delta) .* a2 .* (1 - a2); 
gradStack{1}.W = delta * a1' / M;
gradStack{1}.b = sum(delta, 2) / M;

%% compute weight penalty cost and gradient for non-bias terms
% p_cost = sum(stack{1}.W(:).^2);
% p_cost = p_cost + sum(stack{2}.W(:).^2);
% cost = cost + ei.lambda * p_cost / 2;

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

