% ML_ols - OLS Estimation
% 
% [beta,u,v,esu,r2,espqr]=ML_ols(y,x,det);
%   Inputs:
%       y   = Endogenous Variable (vector)
%       x   = Exogenous Variables (matrix)
%       det = 0 noconstant
%       det = 1 constant
%       det = 2 time trend
%       det = 3 constant + time trend
%   Outputs:
%       beta  = estimated coefficient
%       u     = residuals
%       v     = varcov matrix of the estimates
%       esu   = Residual Variance
%       r2    = R-Squared
%       espqr = estimates standard errors
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)
% This is a modified version of the codes available on Fabio Canova webpage

function [beta,u,v,esu,r2,espar,yhat]=ML_ols(y,x,det,m)
[T, n] = size(x);

cons=ones(T,1); trend=(1:1:T)';
if      det==1; x=[cons x];
elseif  det==2; x=[trend x];
elseif  det==3; x=[cons trend x];
end;
k=size(x,2);                                                                % number of parameters
xx=eye(k)/(x'*x);                                                           % inv(x'x)
beta=xx*x'*y;                                                               % ols coeff
yhat=x*beta;                                                                % fitted values
u=y-yhat;                                                                   % residuals
uu=u'*u;                                                                    % SSR
esu=uu/(T-k);                                                               % Residual Variance
yc=y-mean(y);                                                               % Centered variables
r2=1-(uu)/(yc'*yc);                                                         % R2
% r2c=1-(1-r2)*(T-1)/(T-k);       % adjusted R2


if nargin<4; HAC=0; else HAC=1; end

if HAC==1; % -------------------------------------------------------------- % Estimate HAC standard errors
    Omega0=diag(u.^2);                                                      % For lag(0), the variance estimates are ...
    xOx=(T/(T-k))*x'*Omega0*x;                                              % ... calculated using the White formulation
    if m>0; temp3=0;                                                        % For lag(m),m>0, the variance estimates ...
        for ll=1:m; temp2=0;                                                % are calculated using the Newey–West formulation
            uutl=u(ll+1:T,:).*u(1:T-ll,:);                                  % u_t .* u_{t-l}
            xt=x(ll+1:T,:); xtl=x(1:T-ll,:);                                % x_t  x_{t-l}
            for tt=1:T-ll;
                LagCov=xt(tt,:)'*xtl(tt,:);
                temp=uutl(tt) * (LagCov + LagCov');
                temp2=temp2+temp;
            end
            temp3=(1-(ll/(m+1)))*temp2;
        end
        xOx=xOx+(T/(T-k))*temp3;
    end
    v=xx*xOx*xx;                                                            % varcov matrix of the estimates
else       % -------------------------------------------------------------- % Estimate Normal Standard error
    v=esu*xx;                                                               % varcov matrix of the estimates
end
espar=sqrt(diag(v));                                                        % Standard errors
