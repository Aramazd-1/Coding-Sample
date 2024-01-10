% ML_detrend - OLS detrending
%
% [y, yhat, Beta,sB, J]=ML_detrend(Y);
%   Y - variables to be detrended
%   y - detrended variables 
%   yhat - fitted values
%   Beta - estimated coefficients
%   sB - Standard errors of the Beta 
%   J - abs(tstat>=tinv(.975,T-2))
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function  [y, yhat, Beta]=ML_detrend(Y,jj)
if nargin==1; jj=1; end
[T, N] = size(Y);
cons=ones(T,1); trend=(1:1:T)';

if jj==1;
    x=[cons trend];
else
    x=trend;
    Y=Y-repmat(Y(1,:),T,1);   
end

yhat=NaN(T,N); y=NaN(T,N); Beta=NaN(N,2); sB=NaN(N,2);                      % preallocates
for ii=1:N;
    xx=inv(x'*x);
    beta=xx*x'*Y(:,ii);                                                     % ols estimation
    yhat(:,ii)=x*beta;                                                      % linear trend
    y(:,ii)=Y(:,ii)-yhat(:,ii);                                             % detrended series
    Beta(ii,:)=beta';                                                       % Store estimated coefficients
    sB(ii,:)=sqrt(diag(xx*(y(:,ii)'*y(:,ii))/(T-2)))';
end;
% tstat=Beta(:,2)./sB(:,2);
% J=abs(tstat>=tinv(.975,T-2));