% ML_KalmanSmoother - Kalman Smoother for a Factor Model
% 
%  THE MODEL
%  ¯¯¯¯¯¯¯¯¯
%   X_t = C S_t + e_t,              e_t ~ N(0,R), R is diagonal
%   S_t = A S_{t-1} + u_t,          u_t ~ N(0,Q)
% 
%   S_t|X^{t-1} ~ N( S_{t|t-1} , P_{t|t-1} )
%   X_t|X^{t-1} ~ N( X_{t|t-1} , H_{t|t-1} )
% 
%  _____________
%  THE PROCEDURE
%  ¯¯¯¯¯¯¯¯¯¯¯¯¯
% [xitT,PtT,PtTm]=ML_KalmanSmoother(A,xitt,xittm,Ptt,Pttm,C,R)
% 
% INPUTS:
%       A - the system matrix
%    xitt - S_{t|t}
%   xittm - S_{t|t-1}
%     Ptt - P_{t|t}
%    Pttm - P_{t,t-1|t-1}
%       C - the observation matrix 
%       R - the observation covariance
% OUTPUTS:
%    xitT = S_{t|T}
%     PtT = P_{t|T} 
%    PtTm = P_{t,t-1|T} 
%     

% Matteo Luciani (matteoluciani@yahoo.it)

function [xitT,PtT,PtTm]=ML_KalmanSmoother2(A,xitt,xittm,Ptt,Pttm,C,R)

[T]=size(xitt,2);                                                           % Number of Observations
[N, r]=size(C);                                                             % Number of Variables and Number of States
Pttm=Pttm(:,:,1:end-1);                                                     % P_{t|t-1}, remove the last observation because it has dimension T+1
xittm=xittm(:,1:end-1);                                                     % S_{t|t-1}, remove the last observation because it has dimension T+1
J=zeros(r,r,T);
xitT=[zeros(r,T-1)  xitt(:,T)];                                             % S_{t|T} 
PtT=cat(3,zeros(r,r,T-1),Ptt(:,:,T));                                       % P_{t|T} 
PtTm=zeros(r,r,T);                                                          % P_{t+1|T} 

for jj=1:T-1; 
    if rank(Pttm(:,:,jj+1))<r;
        J(:,:,jj)=Ptt(:,:,jj)*A'*pinv(Pttm(:,:,jj+1));
    else
        J(:,:,jj)=Ptt(:,:,jj)*A'*inv(Pttm(:,:,jj+1));
    end
end;

for jj = 1:T-1;        
    xitT(:,T-jj)= xitt(:,T-jj)+J(:,:,T-jj)*(xitT(:,T+1-jj)-xittm(:,T+1-jj));                      % S_{t|T} 
    PtT(:,:,T-jj)=Ptt(:,:,T-jj)+J(:,:,T-jj)*(PtT(:,:,T+1-jj)-Pttm(:,:,T+1-jj))*J(:,:,T-jj)';    % P_{t|T}         
end
%     j1=find(diag(Pttm(:,:,jj+1))~=0);            
%     J(j1,j1,jj)=Ptt(j1,j1,jj)*A(j1,j1)'*inv(Pttm(j1,j1,jj+1)); 

L=zeros(N,N,T);
K=zeros(r,N,T);

for i=1:T
    L(:,:,i)=inv(C*Pttm(:,:,i)*C'+R);
    K(:,:,i)=Pttm(:,:,i)*C'*L(:,:,i);
end

PtTm(:,:,T)=(eye(r)-K(:,:,T)*C)*A*Ptt(:,:,T-1);                             % P_{t+1|T} 
for j = 1:T-2
    PtTm(:,:,T-j)=Ptt(:,:,T-j)*J(:,:,T-j-1)'+J(:,:,T-j)*(PtTm(:,:,T-j+1)-A*Ptt(:,:,T-j))*J(:,:,T-j-1)';
end

xitT=xitT';