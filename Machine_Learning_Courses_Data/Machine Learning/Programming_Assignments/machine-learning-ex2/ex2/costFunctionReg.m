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

end
