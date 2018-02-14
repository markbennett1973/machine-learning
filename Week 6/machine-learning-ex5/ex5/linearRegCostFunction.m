function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;

% Calculate cost
diff = h - y;
total = sum(diff .* diff);
cost = total / (2 * m);

% remove theta0 from regularization calculation
theta(1) = 0;
reg = sum(theta .* theta) * lambda / (2 * m);

J = cost + reg;

% Calculate gradients
grad = sum(diff .* X) / m;

% add regularization
reg = theta * (lambda/m);

grad = grad + reg';


% =========================================================================

grad = grad(:);

end
