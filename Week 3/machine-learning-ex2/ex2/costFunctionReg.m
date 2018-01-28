function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

% Create a copy of theta for regularization with theta(1) set to zero.
thetaReg = theta;
thetaReg(1) = 0;

% Now calculate J
first = -y .* log(h);
second = (1 - y) .* log(1 - h);
whole = first - second;
J1 = sum(whole) / m;

% Add regularization
squares = thetaReg .* thetaReg;
reg = sum(squares);
reg = reg * lambda / (2 * m);

J = J1 + reg;

% Now compute the gradients

parts = (h - y) .* X;
grad = sum(parts) ./ m;

gradReg = (theta .* (lambda/m))';
gradReg(1) = 0;
grad = grad + gradReg;


% =============================================================

end
