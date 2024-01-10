function  mcmc =...
    BVAR_block_exogenous_covid_adjustment(y,p,lambda,nExo,Tcovid,nCovidParam,N_burn,N_posterior)
% Benjamin Wong
% Monash University
% August 2021
% Estimates a Bayesian with Block exogeniety with the normal indepedent Wishart prior
% with a Minnesota type prior with a COVID correction as outlined by
% Lenza-Primiceri (2021)
% 
% The Bayesian VAR with block exogeniety is estimated using a Gibbs Sampler
% with a Minnesota type prior with a standard Normal-Wishart Prior Structure
% apart from using the COVID correction
%
% The model is estimated without a constant and is identical to the one in
% Kamber and Wong (JIE, 2020)
%
% Order the block exogenous block first with the first nExo variables
% The model is estimated with a MH-within Gibbs algorithm where the MH step
% is used to estimate the COVID parameters
%% INPUTS
%y                  Time series
%p                  number of lags
%lambda             Shrinkage hyper-parameter
%nExo               number of variables in block exogenous block
%Tcovid             index of the first COVID quarter. 2020Q1 in the paper
%nCovidParam    number of quarters to estimate covid parameters.
%N_burn              burn in MCMC
%N_posterior        total posterior draws for MCMC
%% OUTPUTS
% mcmc                      packs all the results

%% Set Priors
% this controls the step size for the MH step
mcmc.MH_const = 5;


%% Preliminaries
[T N] = size(y);

%demean time series
y = y - repmat(mean(y),T,1);
%backcast data
y = [zeros(p,N);y];

K = p*N;        %number of parameters per equation

K_foreign_block = p*nExo; % number of estimated parameters per equation in foreign block
A_post = zeros(K,N); % To store estimated VAR parameters (no cointegration term for now)

small_sig2 =  zeros(N,1);


%Get sigma to set prior based on an AR(4)
for i = 1:N
    [~,small_sig2(i,1),~,~,~] = olsvar(y(:,i),p,'No Constant');
end


%% Set up data in SUR form
Y = y(p+1:end,:);     %Cut Away first p lags (the backcasted stuff)
X = [];

for i = 1:p
    Z = y(p+1-i:end-i,:);
    X = [X Z];
end

X_for = [];

for i = 1:p
    Z = y(p+1-i:end-i,1:nExo);
    X_for = [X_for Z];
end

%% Create Regressors in SUR form for block exogeniety
Z_for = kron(eye(nExo),X_for); Z_dom = kron(eye(N-nExo),X);

Z = [Z_for zeros(size(Z_for,1),size(Z_dom,2));zeros(size(Z_dom,1),size(Z_for,2)) Z_dom];



%% Estimate without COVID data

covid_param0 = abs(randn(nCovidParam,1)*5)+20;

% Maximum likelihood estimation to use the Hessian to condition for the MH
% proposal. This is to just get the hessian and also to get some starting
% values to start the MCMC chain

options.optimisation = optimset('Display','on','TolX',1e-8,'MaxFunEvals',1e10);
covid_param_mle_mode = fminsearch(@(covid_param)loglhVAR_covid(covid_param,Y,Z,Tcovid),covid_param0,options.optimisation)

[logLH,betahat,sigmahat]=loglhVAR_covid(covid_param_mle_mode,Y,Z,Tcovid);


fun = @(covid_param) loglhVAR_covid(covid_param,Y,Z,Tcovid);
Hess = hessian(fun,covid_param_mle_mode);
HH=inv(Hess);

[V,E] = eig(HH);

%% Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
v_prior = K+1;
S_prior = (K+1)*diag(small_sig2);


%% Set Up Prior on VAR coefficients
   V_prior = [];
    for jj = 1:nExo
        for ii = 1:p
            V_prior = [V_prior;repmat((lambda^2)*small_sig2(jj)/(ii^2),nExo,1)./small_sig2(1:nExo,1)];
        end
    end
    
    for jj = nExo+1:N
        for ii = 1:p
            V_prior = [V_prior;repmat((lambda^2)*small_sig2(jj)/(ii^2),N,1)./small_sig2];
        end
    end
    V_prior = diag(V_prior);

%% Initialize Bayesian posterior parameters using OLS values
alpha_draw = betahat;     % This is the single draw from the posterior of alpha
SIGMA_draw = sigmahat+diag(small_sig2); % This is the single draw from the posterior of SIGMA
hyper_draw = covid_param_mle_mode;

%% Gibbs sampler
n_eta = nCovidParam;
% Storage space for posterior draws


mcmc.beta = NaN(size(betahat,1),N_posterior);
mcmc.sigma = NaN(N,N,N_posterior);
mcmc.hyper = NaN(nCovidParam,N_posterior);
mcmc.logML = NaN(N_posterior,1);

mcmc.MH_accept = zeros(N_burn+N_posterior,1);

tic




%% MCMC
for kk = 1:N_burn+N_posterior
    
    
    %% draw alpha

    eta_draw=hyper_draw;

     invweights = ones(T,1);
    
    invweights(Tcovid:Tcovid+n_eta-1)=eta_draw;

    y = diag(1./invweights)*Y;
    x = repmat(1./invweights,N,size(Z,2)).*Z;
    
    a_prior = zeros(K_foreign_block*nExo+K*(N-nExo),1);

    VARIANCE = kron(inv(SIGMA_draw),eye(T));
    V_post = inv(inv(V_prior) + x'*VARIANCE*x);
    
    [eigenvector,eigenvalue] = eig(V_post);
    
        if sum(diag(eigenvalue) < 0) > 0
            for jj = 1:size(eigenvalue,1)
                if eigenvalue(jj,jj) < 0
                    eigenvalue(jj,jj) = 1e-8;
                end
            end
            V_post = eigenvector*eigenvalue*eigenvector';
        end
    
     a_posterior = V_post*(inv(V_prior)*a_prior + x'*VARIANCE*y(:));
    alpha_draw = a_posterior + chol(V_post,'lower')*randn(size(Z,2),1); % Draw of alpha
    
    
    %% Draw SIGMA
    % Posterior of SIGMA|ALPHA,Data ~ iW(inv(S_post),v_post)
    
    resid = reshape(y(:) - x*alpha_draw,T,N);
    SSE = resid'*resid;
    
    v_post = T + v_prior;
    S_post = S_prior + SSE;%(Y - X*ALPHA_draw)'*(Y - X*ALPHA_draw);
    
    SIGMA_draw = inv(wishrnd(inv(S_post),v_post));% Draw SIGMA
    %% Draw s_t with a MH step
    hyper_proposal = mvnrnd(hyper_draw',HH*(mcmc.MH_const)^2,1)';

    
    logMLnew =...
        logML_covid_Block_exo(hyper_proposal,alpha_draw,SIGMA_draw,V_prior,S_prior,Y,Z,nExo,p,Tcovid);
    
    logMLold =...
        logML_covid_Block_exo(hyper_draw,alpha_draw,SIGMA_draw,V_prior,S_prior,Y,Z,nExo,p,Tcovid);
       
    
    if logMLnew>logMLold
        hyper_draw = hyper_proposal;
        mcmc.MH_accept(kk) = 1;
    else
        if rand(1)<exp(logMLnew-logMLold);
            hyper_draw = hyper_proposal;
            mcmc.MH_accept(kk) = 1;
        end
    end
    
    %%
    if kk >N_burn

        
        mcmc.beta(:,kk-N_burn) = alpha_draw;
        mcmc.sigma(:,:,kk-N_burn) = SIGMA_draw;
        mcmc.hyper(:,kk-N_burn) = hyper_draw;
        
        mcmc.logML(kk-N_burn,1) = ...
            logML_covid_Block_exo(hyper_draw,alpha_draw,SIGMA_draw,V_prior,S_prior,Y,Z,nExo,p,Tcovid);
        

    end
    
    if mod(kk,1000) == 0
        disp([num2str(kk),' MCMC draws done',''])
        disp(['MCMC Acceptance Rate: ',num2str(100*mean(mcmc.MH_accept(1:kk))),'%'])
        disp(['MCMC ',num2str(toc),' seconds elapsed'])
        
        hyper_draw'
        
        
    end
    
    
    
end

end