function [f,g] = softmax_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m = size(X,2);
  n = size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta = reshape(theta, n, []);
  theta = [theta zeros(n,1)];
  num_classes = size(theta, 2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  % 1st version
%     for j=1:m
%       k_sum = sum(exp(theta' * X(:,j)));  
%       for k=1:num_classes
%         p = exp(theta(:,k)' * X(:,j)) / k_sum;
%         f = f - (y(j) == k) * log(p);
%         g(:,k) = g(:,k) - X(:,j) * ((y(j) == k) - p);
%       end   
%     end

  % 2nd version
%   for j=1:m  
%     indicators = zeros(num_classes,1);
%     indicators(y(j)) = 1; % K * 1
%     p = exp(theta' * X(:,j));
%     p = p / sum(p); % K * 1
%     f = f - indicators' * log(p); % 1 * 1
%     g = g - X(:,j) * (indicators - p)'; % n * K
%   end

  % 3rd version
  ind = zeros(num_classes, m); % K * m
  tmp = sub2ind(size(ind), y, 1:m);
  ind(tmp) = 1;
  
  p = exp(theta' * X);
  p = bsxfun(@rdivide, p, sum(p)); % K * m
  diff_ind_p = ind - p; % K * m
  
  f = -sum(sum(ind .* p, 2));
  g = -X * diff_ind_p';
  
  g = g(:,1:end-1);
  g = g(:); % make gradient a vector for minFunc
end
