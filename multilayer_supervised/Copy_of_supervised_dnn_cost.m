function [ cost, grad, pred_prob] = supervised_dnn_cost(theta, ei, data, labels, pred_only)
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
a1 = data; % 784*60000
M = size(data, 2);
% hidden layer
W1 = stack{1}.W; % 256*784
b1 = stack{1}.b; % 256*1
a2 = zeros(size(b1,1), M); % 256*60000
for m=1:M
  a2(:,m) = sigmoid(W1 * a1(:,m) + b1);
end
% output layer
W2 = stack{2}.W; % 10*256
b2 = stack{2}.b;
a3 = zeros(size(b2,1), M); % 10*60000
for m=1:M
  a3(:,m) = sigmoid(W2 * a2(:,m) + b2);
end
pred_prob = a3;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
deltaW1 = zeros(size(W1));
deltab1 = zeros(size(b1));
deltaW2 = zeros(size(W2));
deltab2 = zeros(size(b2));
for m=1:M
  % output layer
  y = zeros(size(a3,1), 1);
  y(labels(m)) = 1;
  delta3 = -(y - a3(:,m)) .* (a3(:,m) .* (1 - a3(:,m))); % 10*1
  deltaW2 = deltaW2 + delta3 * a2(:,m)';
  deltab2 = deltab2 + delta3;
  % hidden layer
  delta2 = W2' * delta3 .* (a2(:,m) .* (1 - a2(:,m))); % 256*1
  deltaW1 = deltaW1 + delta2 * a1(:,m)'; % 256*784
  deltab1 = deltab1 + delta2; % 256*1
end

%% compute gradients using backpropagation
lambda = 0;
% input layer
gradStack{1}.W = deltaW1 / M + lambda * W1;
gradStack{1}.b = deltab1 / M;
% hidden layer
gradStack{2}.W = deltaW2 / M + lambda * W2;
gradStack{2}.b = deltab2 / M;

%% compute weight penalty cost and gradient for non-bias terms
cost = 0;
for m=1:M
  cost = cost - (1 - a3(labels(m),m));
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

