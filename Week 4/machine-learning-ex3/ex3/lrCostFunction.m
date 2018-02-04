function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


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

grad = grad(:);

end
