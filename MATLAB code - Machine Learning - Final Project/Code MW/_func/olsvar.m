%Benjamin Wong
%RBNZ
%Nov 2014

function [A,SIGMA,U,invXX,X] = olsvar(y,p,nc)
% This estimates an OLS reduced form VAR
%
%INPUTS
%y  Data(TxN)
%p  lags
%nc put some arbiturary number if you don't want a constant

%OUTPUTS
%A      VAR Coefficients
%SIGMA  Reduced form variance covariance matrix
%U      Time Series of Residuals

T = size(y,1)-p;      %length of Time Series

Y = y(p+1:end,:);     %Cut Away first p lags

if nargin == 2
    X = ones(T,1);
else
    X = [];
end

for i = 1:p
    Z = y(p+1-i:end-i,:);
    X = [X Z];
end

%Get OLS coefficients
A = (X'*X)\(X'*Y);

%Get OLS residuals
U = Y-X*A;

%Get OLS variance covariance matrix
%dof is Total Sample-lags-Number of Regressors
SIGMA   = (U'*U)/(T-size(A,1));

invXX   = inv(X'*X);

end

