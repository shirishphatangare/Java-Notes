clear ; close all; clc

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
m = size(X, 1);
p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly); 
X_poly = [ones(m, 1), X_poly]; 
                  
% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];  
       
% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];   


% Train model with lambda = 3
lambda = 3;
theta = trainLinearReg(X_poly, y, lambda);

% Calculate cost with lambda = 0 ---- WHY lambda 0 here and NOT 3 ??
lambda = 0;
error_train = linearRegCostFunction(X_poly, y, theta, lambda);
error_val = linearRegCostFunction( X_poly_val, yval, theta, lambda);
error_test = linearRegCostFunction( X_poly_test, ytest, theta, lambda);
	
fprintf("error_train = %f\n",error_train);
fprintf("error_val = %f\n",error_val);
fprintf("error_test = %f\n",error_test);