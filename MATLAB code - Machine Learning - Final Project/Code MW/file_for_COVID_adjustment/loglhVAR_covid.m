function [logLH,betahat,sigmahat]=loglhVAR_covid(covid_param,Y,Z,Tcovid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the concentrated log-likelihood of a VAR,
% augmented with a change in volatility at the time of Covid (March 2020)
%
% Last modified: 07/30/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ncp=4;      % number of covid hyperparameters
% eta=MIN.eta'+(MAX.eta'-MIN.eta')./(1+exp(-par));
% 
% invweights = ones(T,1);     % vector of s_t
% invweights(Tcovid)=eta(1);
% invweights(Tcovid+1)=eta(2);
% if T>Tcovid+1;
%     invweights(Tcovid+2:T)=1+(eta(3)-1)*eta(4).^[0:T-Tcovid-2];
% end
% y = diag(1./invweights)*y;
% x = diag(1./invweights)*x;

[T N] = size(Y);

%eta = covid_param(1:end-1,1);
eta = covid_param;
n_eta = size(eta,1);

%rho = covid_param(end,1);


 invweights = ones(T,1); 
    invweights(Tcovid:Tcovid+n_eta-1)=eta;
    %invweights(Tcovid+1)=eta(2);
    %if T>Tcovid+1;
    %    invweights(Tcovid+n_eta:T)=1+(eta(end,1)-1)*rho.^[1:T-Tcovid-n_eta+1];
    %end
    Y = diag(1./invweights)*Y;
    x = repmat(1./invweights,N,size(Z,2)).*Z;
    

%% output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MLE of the VAR coefficients (it is a function of invweights)
betahat=(x'*x)\(x'*Y(:));

% residuals at the MLE estimates of the VAR coefficients
epshat=reshape(Y(:)-x*betahat,T,N);

% MLE of VAR residual covariance matrix
sigmahat=epshat'*epshat/T;

% concentrated log-LH
logLH=-.5*T*log(det(sigmahat))- N*sum(log(invweights));

if sum(eta < 1) > 0 %|| rho < 0
    logLH = -1e15;
end

logLH=-logLH;       % switch sign because we are minimizing