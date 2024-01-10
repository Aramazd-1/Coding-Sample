function logML=logML_covid_Block_exo(hyperparam,alpha,SIGMA,V_prior,S_prior,Y,Z,nExo,p,Tcovid);

%[logML,betadraw,drawSIGMA]=logML_covid_Block_exo(par,y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,draw,hyperpriors,priorcoef,Tcovid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function computes the log-posterior (or the logML if hyperpriors=0),
% and draws from the posterior distribution of the coefficients and of the
% covariance matrix of the residuals of the BVAR of Giannone, Lenza and
% Primiceri (2015), augmented with a change in volatility at the time of
% Covid (March 2020).
%
% Last modified: 06/02/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T N] = size(Y);

K = p*N;

%% hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eta=hyperparam;


n_eta = size(eta,1);


invweights = ones(T,1);

invweights(Tcovid:Tcovid+n_eta-1)=eta;

y = diag(1./invweights)*Y;
x = repmat(1./invweights,N,size(Z,2)).*Z;

%  %% Set up BVAR prior
%  %Get sigma to set prior based on an AR(4)
for i = 1:N
    [~,small_sig2(i,1),~,~,~] = olsvar(Y(:,i),p,'No Constant');

end



%% Set up the likelihood

resid = y(:) - x*alpha;

logML = - 0.5 *(resid'*kron(eye(T),inv(SIGMA))*resid) -(T/2)*log(det(SIGMA));

% Add the likelihood from the weights
logML = logML - N*sum(log(invweights));

% Add in log Prior of VAR coefficients
logML = logML - 0.5*(alpha'*inv(V_prior)*alpha);% - 0.5*log(prod(diag(V_prior)));

% Add in log prior of the covariance matrix
logML = logML - 0.5* trace(inv(SIGMA)*inv((K+1)*diag(small_sig2))) - 0.5*(1+N+K+1)*det(SIGMA);

% add in log of the hyperpriors

logML = logML - sum(2*log(eta));


if any(eta<1)

    logML = -1e20;

end

end

function r=logGammapdf(x,k,theta);
r=(k-1)*log(x)-x/theta-k*log(theta)-gammaln(k);
end
function r=logBetapdf(x,al,bet);
r=x^(al-1)*(1-x)^(bet-1)/beta(al,bet);
end
function r=logIG2pdf(x,alpha,beta);
r=alpha*log(beta)-(alpha+1)*log(x)-beta./x-gammaln(alpha);
end
function C = cholred(S);
[v,d] = eig((S+S')/2);
d = diag(real(d));
warning off
scale = mean(diag(S))*1e-12;
J = (d>scale);
C = zeros(size(S));
C(J,:) = (v(:,J)*(diag(d(J)))^(1/2))';
end