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

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
valCount = length(vals);

errors = zeros(valCount, valCount);

for cIndex = 1:valCount
	cPotential = vals(cIndex);
	for sIndex = 1:valCount
    fprintf(['Testing cIndex %d, sIndex %d\n'], cIndex, sIndex);
		sPotential = vals(sIndex);
    
    model= svmTrain(X, y, cPotential, @(x1, x2) gaussianKernel(x1, x2, sPotential)); 
		predict = svmPredict(model, Xval);
    error = mean(double(predict ~= yval));
		errors(cIndex, sIndex) = error;
  end
end
  
% find row, col of min errors
[minVal, sIndex] = min(min(errors, [], 1));
[minVal, cIndex] = min(min(errors, [], 2));

C = vals(cIndex);
sigma = vals(sIndex);

    fprintf(['Got optimum C %f, sigma %f\n'], C, sigma);




% =========================================================================

end
