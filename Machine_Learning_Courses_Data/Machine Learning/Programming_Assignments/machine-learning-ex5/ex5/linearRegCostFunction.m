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

% Regularized linear regression cost function
H = X * theta;
C = (1 / (2 * m)) * ((H - y)' * (H - y));

T = theta(2:end,:);
R = (lambda / (2 * m)) * (T' * T);

J = C + R;

% Regularized linear regression gradient
G = (1 / m) * (X' * (H - y));
R = (lambda / m) * T;

grad1 = G(1,:);
GR = G(2:end,:) .+ R;

grad = [grad1; GR];

% =========================================================================

grad = grad(:);

end
