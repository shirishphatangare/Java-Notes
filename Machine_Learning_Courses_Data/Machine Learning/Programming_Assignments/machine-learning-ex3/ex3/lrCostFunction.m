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


T = X * theta;
H = 1 ./ (1 + exp(-1 * T));
J1 = ((y' * log(H)) + ((1-y)' * log(1-H))) * (-1/m);

Q = theta([2:end],:);
J2 = (lambda / (2 * m)) * (Q' * Q);
J = J1 + J2;

T1 = X(:,1);
G1 = (T1' * (H - y)) * (1/m);

T2 = X(:,[2:end]);
R = (lambda / m) * Q;
G2 = (T2' * (H - y)) * (1/m);
G3 = G2 + R;
grad=[G1;G3];


% =============================================================

grad = grad(:);

end
