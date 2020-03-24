function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%Collaborative Filtering Cost Function without regularization

predictionsY = X * Theta';
M = predictionsY - Y;
T = M .^ 2;
unregularizedJ = (1/2) * sum(sum(R .* T)); % The sum of all the elements of T for which the corresponding element in R equals 1.

% Regularized cost function
regularizedX = (lambda / 2) * sum(sum(X .^ 2));
regularizedTheta = (lambda / 2) * sum(sum(Theta .^ 2));
J = unregularizedJ + regularizedX + regularizedTheta;


% Collaborative Filtering Gradient
% Looping over movies to compute gradient for each movie

for i = 1:num_movies
	idx = find(R(i,:) == 1); % For movie i find all columns(users) with rating = yes i.e find all users who have rated movie i
	ThetaTemp = Theta(idx,:); % Find theta parameters only for users who have rated movie i(idx)
	YTemp = Y(i,idx); % Get actual ratings for movie i by all users who have rated movie i(idx)
	
	% Now using ThetaTemp and YTemp, calculate gradient for movie i
	X_grad(i,:) = (X(i,:) * ThetaTemp' - YTemp) * ThetaTemp; % size of this matrix will be 1 * k(no. of features of movie i)
end

% Looping over users to compute gradient for each user

for j = 1:num_users
	jdx = find(R(:,j) == 1); % For user j find all movies he has rated
	XTemp = X(jdx,:); % Find movie features only for movies user j has rated (jdx)
	YTemp = Y(jdx,j); % Get actual ratings for movies all which user j has rated
	
	% Now using XTemp and YTemp, calculate gradient for each user j
	Theta_grad(j,:) = (XTemp * Theta(j,:)' - YTemp)' * XTemp; % size of this matrix will be 1 * k(no. of parameters of user j)
end

% Regularized Gradient
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
