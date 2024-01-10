% ML_adf - Augmented Dickey Fuller Test
%
% --------------------------------------------------------------------
% The Model:
% delta[x(t)] = DET + phi*x(t-1) + psi(1)*delta[x(t-1)] + ... +
% psi(k)*delta[x(t-k-1)] + u(t)
%
% [adf, k]=ML_adf(y,p,kmax);
% Input:
%       y variable to be tested
%       p = -1 no deterministic part
%       p = 0 then DET = constant
%       p = 1 then DET = constant & trend
%       kmax = maximum lag of the autoregressive process (optional)
% Output:
%       adf = test statistic
%       k-1 = order of the autoregressive process 
%       (automatically selected with the Bayesan Information Criteria)
% ********************************************************************
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function [adf, k, gamma, s_gamma]=ML_adf(y,p,kmax)

T=size(y,1);
if nargin<3; kmax=ceil(12*(T/100)^(1/4)); end;
constant=ones(T-1,1);
trend=(1:1:T-1)';
alpha=zeros(kmax+p+2,kmax+1);
dy=y(2:T)-y(1:T-1); 
t=size(dy,1);
%X=zeros(t,kmax); % matrix containing the delta for the regression

% ----------------------------
% Augmented Dickey Fuller Test
% ----------------------------
if kmax>1;
    tstat=zeros(1,kmax+1);
    bic=zeros(1,kmax+1);    
    for ii=1:1:kmax;
        X=zeros(t-ii,ii); % matrix containing the delta for the regression
        for jj=1:ii;
            X(:,jj)=dy(ii-jj+1:t-jj);
        end;
        if p==0; DET=constant;         reg = [y(1+ii:T-1) DET(1:T-ii-1,:) X]; end;
        if p==1; DET=[constant trend]; reg = [y(1+ii:T-1) DET(1:T-ii-1,:) X]; end;
        if p==-1;                      reg = [y(1+ii:T-1) X];                 end;        
        alpha(1:ii+p+2,ii+1)= inv(reg'*reg)*reg'*dy(1+ii:t);
        e=dy(1+ii:t)-reg*alpha(1:ii+p+2,ii+1);
        sig2=e'*e/(size(dy,1)-size(reg,2)-1);
        xx=inv(reg'*reg);
        s_alpha(ii+1)=sqrt(sig2*xx(1,1));
        tstat(ii+1)=alpha(1,ii+1)/s_alpha(1,ii+1);    
        bic(ii+1) = log(e'*e/t) + (size(reg,2)/t)*log(t);
        clear e reg xx
    end;    
end;

% ------------------
% Dickey Fuller Test
% ------------------
if p==0; DET=constant;         reg = [y(1:T-1) DET(1:T-1,:)]; end;
if p==1; DET=[constant trend]; reg = [y(1:T-1) DET(1:T-1,:)]; end;
if p==-1;                      reg = [y(1:T-1)];           end;
alpha(1:p+2,1)=inv(reg'*reg)*reg'*dy;
e=dy-reg*alpha(1:p+2,1);
sig2(1)=e'*e/(size(dy,1)-size(reg,2)-1);
xx=inv(reg'*reg);
s_alpha(1)=sqrt(sig2*xx(1,1));
tstat(1)=alpha(1,1)/sqrt(sig2(1)*xx(1,1));    
bic(1) = log(e'*e/t) + (size(reg,2)/t)*log(t);

k=0;
adf=0;
jj=ML_argmin(bic');
k=jj-1; adf=tstat(jj); gamma=alpha(1:jj+p+1,jj);;s_gamma=(s_alpha(jj));

%alpha
%s_alpha
%for ii=1:kmax+1;    if bic(ii)==min(bic); k=ii-1; adf=tstat(ii);  end; end;
