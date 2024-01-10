function  results =...
    BVAR_block_exogenous(y,p,lambda,nExo,nburn,total_draws)
% Benjamin Wong
% Monash University
% August 2021
% Estimates a Bayesian with Block exogeniety with the normal indepedent Wishart prior
% with a Minnesota type prior
% Because the regressors differ on the RHS due to block exogneity, the
% prior is not natural conjugate and posterior simulation is done with a
% Gibbs sampler
%
% The Bayesian VAR with block exogeniety is estimated using a Gibbs Sampler
% with a Minnesota type prior with a standard Normal-Wishart Prior Structure
%
% The model is estimated without a constant and is identical to the one in
% Kamber and Wong (JIE, 2020)
%
% Order the block exogenous block first with the first nExo variables
%
%% INPUTS
%
%y                  Time series
%p                  number of lags
%lambda             Shrinkage hyper-parameter (see Banbura, Giannone and Reichlin, JAE, 2010)
%nExo               number of variables in block exogenous block
%nburn              burn in MCMC
%total_draws        total draws for MCMC
%% OUTPUTS
% all results are packed into the class results. The results class is written in a way so that this can be
% fed into the function to use the BN decomposition

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

%% Set up Prior

a_prior = zeros(K_foreign_block*nExo+K*(N-nExo),1);

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


%% Create Regressors
Z_for = kron(eye(nExo),X_for); Z_dom = kron(eye(N-nExo),X);

Z = [Z_for zeros(size(Z_for,1),size(Z_dom,2));zeros(size(Z_dom,1),size(Z_for,2)) Z_dom];


%%

% ML estimators
A_OLS_foreign_block = (X_for'*X_for)\(X_for'*Y(:,1:nExo));
A_OLS_domestic_block = (X'*X)\(X'*Y(:,nExo+1:N));
a_OLS = [A_OLS_foreign_block(:);A_OLS_domestic_block(:)]; % This is the vector of parameters, i.e. it holds
% that a_OLS = vec(A_OLS)
Resid_ols_foreign = Y(:,1:nExo) - X_for*A_OLS_foreign_block;
Resid_ols_domestic = Y(:,nExo+1:N) - X*A_OLS_domestic_block;
ols_resid = [Resid_ols_foreign Resid_ols_domestic];
SSE = ols_resid'*ols_resid;   % Sum of squared errors
SIGMA_OLS = SSE./(T);


% Hyperparameters on inv(SIGMA) ~ W(v_prior,inv(S_prior))
v_prior = (K+1);
S_prior = (K+1)*diag(small_sig2);


%% Initialize Bayesian posterior parameters using OLS values
alpha = a_OLS;     % This is the single draw from the posterior of alpha
SIGMA_draw = SIGMA_OLS+(K+1)*diag(small_sig2); % This is the single draw from the posterior of SIGMA

%% Gibbs sampler

% Storage space for posterior draws
num_MCMC_posterior_taken = 0;

results.A_post = zeros(K,N,total_draws-nburn);
results.SIGMA_post = zeros(N,N,total_draws-nburn);

%% MCMC
for jj = 1:total_draws %10000 MCMC draws, burn first 2500


    VARIANCE = kron(inv(SIGMA_draw),eye(T));
    V_post = inv(inv(V_prior) + Z'*VARIANCE*Z);
    a_posterior = V_post*(inv(V_prior)*a_prior + Z'*VARIANCE*Y(:));
    alpha = a_posterior + chol(V_post,'lower')*randn(size(Z,2),1); % Draw of alpha



    alpha_for = alpha(1:K_foreign_block*nExo);
    alpha_dom = alpha(K_foreign_block*nExo+1:end);

    ALPHA = reshape(alpha_dom,[],N-nExo); % Draw of ALPHA

    ALPHA_for_temp = reshape(alpha_for,[],nExo);
    ALPHA_for = [];
    for ii = 1:p
        ALPHA_for = ...
            [ALPHA_for;ALPHA_for_temp((ii-1)*nExo+1:ii*nExo,:);zeros(N-nExo,nExo)];

    end

    ALPHA_draw = [ALPHA_for ALPHA];
    % Posterior of SIGMA|ALPHA,Data ~ iW(inv(S_post),v_post)
    v_post = T + v_prior;
    S_post = S_prior + (Y - X*ALPHA_draw)'*(Y - X*ALPHA_draw);


    SIGMA_draw = inv(wishrnd(inv(S_post),v_post));% Draw SIGMA



    if jj >nburn
        num_MCMC_posterior_taken = num_MCMC_posterior_taken+1;

        results.A_post(:,:,num_MCMC_posterior_taken) = ALPHA_draw;
        results.SIGMA_post(:,:,num_MCMC_posterior_taken) = SIGMA_draw;

    end

end

end