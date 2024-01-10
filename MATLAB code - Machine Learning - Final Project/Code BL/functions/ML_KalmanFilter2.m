% ML_KalmanFilter2 - Kalman Filter Estimation of a Factor Model
%  _________
%  THE MODEL
%  ¯¯¯¯¯¯¯¯¯
%   X_t = beta1 + beta2*t + C S_t + e_t,        e_t ~ N(0,R), R is diagonal
%   S_t = mu + A S_{t-1} + u_t,                 u_t ~ N(0,Q)
% 
%       S_t|X^{t-1} ~ N( S_{t|t-1} , P_{t|t-1} )
%       X_t|X^{t-1} ~ N( X_{t|t-1} , H_{t|t-1} )
% 
%  _____________
%  THE PROCEDURE
%  ¯¯¯¯¯¯¯¯¯¯¯¯¯
% [xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter(initx,initV,x,A,C,R,Q,mu,beta)
% 
% INPUTS:
%   x - the data
%   C - the observation matrix 
%   A - the system matrix
%   Q - the system covariance 
%   R - the observation covariance
%   initx - the initial state (column) vector 
%   initV - the initial state covariance 
%   mu   - constant in transition equation (optional)
%   beta - constant and linear trend in observation equation (optional)
% OUTPUTS:
%   xittm = S_{t|t-1}
%    Pttm = P_{t|t-1}
%    xitt = S_{t|t} 
%     Ptt = P_{t|t}
%  loglik = value of the loglikelihood
% 

% Matteo Luciani (matteoluciani@yahoo.it)

function [xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter2(initx,initV,x,A,C,R,Q,mu,beta)

T=size(x,1);                                                                % Number of Observations
r=size(A,1);                                                                % Number of states
xittm=[initx zeros(r,T)];                                                   % S_{t|t-1}
xitt=zeros(r,T);                                                            % S_{t|t} 
Pttm=cat(3,initV,zeros(r,r,T));                                             % P_{t|t-1}
Ptt=zeros(r,r,T);                                                           % P_{t|t}
loglik=0;                                                                   % Initialize the log-likelihood
y=x';                                                                       % transpose for convenience

if nargin<8; mu=zeros(r,1); end
if nargin<9; beta=zeros(1,2); end

for j=1:T;
    
    %%% _____________ %%%
    %%% Updating Step %%%
    %%% ¯¯¯¯¯¯¯¯¯¯¯¯¯ %%%   
    X=C*xittm(:,j) + beta*[1;j];                                            % X_{t|t-1} - Prediction
    H=C*Pttm(:,:,j)*C'+R;                                                   % H_{t|t-1} - Conditional Variance of the Observable   
    Hinv = inv(H);    
    e = y(:,j) - X;                                                         % error (innovation)
    xitt(:,j)=xittm(:,j)+Pttm(:,:,j)*C'*Hinv*e;                             % S_{t|t}
    Ptt(:,:,j)=Pttm(:,:,j)-Pttm(:,:,j)*C'*Hinv*C*Pttm(:,:,j);               % P_{t|t}   

    %%% _______________ %%%
    %%% Prediction Step %%%
    %%% ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯ %%%
    xittm(:,j+1)=mu+A*xitt(:,j);                                            % S_{t|t-1} - States
    Pttm(:,:,j+1)=A*Ptt(:,:,j)*A'+Q;                                        % P_{t|t-1} - Conditional Variance of the State 
    loglik = loglik + 0.5*(log(det(Hinv))  - e'*Hinv*e);                    % Log Likelihood   
    
end

xitt=xitt';
xittm=xittm';

% Io se guardo tipo kim and nelson (pag.26) o Lutkepohl (pag.632) lo scriverei cosi:
%   F  = Z_t*PZ + R_t;
%   iF  = inv(F);
%   V   = y_t - Z_t*A;
%   S.loglik = S.loglik + 0.5*(-log(det(F))  - V'*iF*V);
% 
% Nota pero' che:
%   det(iF)=1/det(F)
%   log(det(iF))=log(1)-log(det(F))
%   log(det(iF))= -log(det(F))
% 


