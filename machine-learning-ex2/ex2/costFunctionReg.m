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

% Compute the cost Function
[J, grad] = costFunction(theta, X, y);

% Theta(1) should not be regularized 
theta_one = theta(1);
theta = theta(2:end);

% Compute the regularized gradient function
grad = grad + [0; lambda/m*theta];

% Compute the regularized function
reg = lambda/(2*m)*theta'*theta;

% Compute the regularized cost function
J = J + reg;


% =============================================================

end
