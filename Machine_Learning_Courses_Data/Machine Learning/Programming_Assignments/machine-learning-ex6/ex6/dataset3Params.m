function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


CArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
SArray = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
predictionErrorCount = 1;
counter = 1;

for i = 1 : size(CArray,2)
	for j = 1 : size(SArray,2)
		model = svmTrain(X, y, CArray(1,i), @(x1, x2) gaussianKernel(x1, x2, SArray(1,j)));
		predictions = svmPredict(model, Xval);
		predictionError(1,predictionErrorCount) = mean(double(predictions ~= yval));
		predictionErrorCount = predictionErrorCount + 1;
	end
end

[value,index] = min(predictionError,[],2);

for i = 1 : size(CArray,2)
	for j = 1 : size(SArray,2)
		if (index == counter)
			C = CArray(i);
			sigma = SArray(j);
		end	
		counter = counter + 1;
	end
end

% Determined the best C and sigma parameters as below by training on the training set, and measuring the validation error on the validation set. These C and sigma values are used directly in script ex6 instead of calling this function. Calling function will be slow as it will train 64 models every time.

%C = 1;
%sigma = 0.1;

% =========================================================================

end
