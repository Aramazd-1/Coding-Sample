% ML_VAR - Estimates a VAR(k) for a vector of variables y
% if y is a single variable it estimates an AR model with OLS
%
% [A,u,AL,CL]=ML_VAR(y,det,jj);
%   y  = vector of endogenous variables
%   k  = number of lags
%   det = 0 noconstant
%   det = 1 constant
%   det = 2 time trend
%   det = 3 constant + time trend
%   A  = Matrix of coefficients for the reduced form 
%   u  = Vector of Residuals
%
%  y_t=A(L)y_{t-1}+u_t
%  y_t=C(L)u_t
% ----------------------------------------------------------

% written by Matteo Luciani (matteoluciani@yahoo.it)

function [A,u,AL,CL,C1]=ML_VAR(y,k,det,s)
[T, N] = size(y);

%%% Building Up the vector for OLS regression %%%
yy=y(k+1:T,:);
xx=NaN(T-k,N*k);
for ii=1:N    
    for jj=1:k
        xx(:,k*(ii-1)+jj)=y(k+1-jj:T-jj,ii);
    end
end

%%% OLS Equation-By-Equation %%%
if det==0; ll=0; elseif  det==3; ll=2; else; ll=1; end
A=NaN(N*k+ll,N); u=NaN*yy;
for ii=1:N
    [A(:,ii),u(:,ii)]=ML_ols(yy(:,ii),xx,det);
end

At=A; if det==3; At(1:2,:)=[]; elseif det==1||det==2; At(1,:)=[]; end
AL=NaN(N,N,k); for kk=1:k; AL(:,:,kk)=At(1+kk-1:k:end,:)'; end  


%%% Impulse Responses %%%
if nargin<4;s=20; end
CL(:,:,1) = eye(N);
for ss=2:s
    CL(:,:,ss) = 0;
    for ii = 1:min(ss-1,k)        
        temp3=AL(:,:,ii)*CL(:,:,ss-ii);        
        CL(:,:,ss)=CL(:,:,ss)+temp3;
    end
end

C1=eye(N); for ii=1:k; C1=C1-AL(:,:,ii); end; C1=inv(C1);                     % Long Run Multipliers


