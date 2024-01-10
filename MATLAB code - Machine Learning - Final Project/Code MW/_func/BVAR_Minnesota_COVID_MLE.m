function results =...
    BVAR_Minnesota_COVID_MLE(y,p,lambda,Tcovid,nCovidParam)
% Benjamin Wong
% Monash University
% August 2021
% Estimates large BVAR by using the Normal-Wishart version of the Minnesota
% Prior through dummy observations
%The Lenza-Primiceri COVID correction is implemented
% estimating s_t using MLE as described in the paper, then reweighting and
% estimating the model as a standard Bayesian VAR.
%
%
%% INPUTS
%
%y                  Time series
%p                  number of lags
%lambda             Shrinkage hyper-parameter (see Banbura, Giannone and Reichlin, JAE, 2010)

%% OUTPUTS
% all results are packed into the class results. The estimated VAR
% parameters across the chain and the MLE estimated Covid parameters can be
% found there. The results class is written in a way so that this can be
% fed into the function to use the BN decomposition

%% Preliminaries

[T N] = size(y);

%demean time series
y = y - repmat(mean(y),T,1);
%backcast data
y = [zeros(p,N);y];

K = p*N;        %number of parameters per equation

small_sig2 =  zeros(N,1);
Stack_AR_coeff = zeros(p,N);

for i = 1:N
    %     calculate variance for each equation to set prior
    %     by fitting an AR(4) per equation
    [~,small_sig2(i,1),~,~,~] = olsvar(y(:,i),p,1);
end

%Create Data Matrices
Y = y(p+1:end,:);     %Cut Away first p lags (the backcasted stuff)
X = [];

for i = 1:p
    Z = y(p+1-i:end-i,:);
    X = [X Z];
end

%% Estimate COVID parameters here
covid_param0 = abs(randn(nCovidParam,1)*5)+20;

Z = kron(eye(N),X);

% Maximum likelihood estimation to use the Hessian to condition for the MH
% proposal

options.optimisation = optimset('Display','on','TolX',1e-8,'MaxFunEvals',1e10);
covid_param_mle_mode = fminsearch(@(covid_param)loglhVAR_covid(covid_param,Y,Z,Tcovid),covid_param0,options.optimisation)

%% Do covid scaling here

invweights = ones(T,1);
invweights(Tcovid:Tcovid+nCovidParam-1)=covid_param_mle_mode;


Y = repmat(1./invweights,1,N).*Y;
X = repmat(1./invweights,1,N*p).*X;


%% Set up dummy observations


Y_d = [zeros(N*p,N);
    diag(sqrt(small_sig2))];

for i = 1:p
    Y_d((i-1)*N+1:i*N,:) = diag(Stack_AR_coeff(i,:)'.*repmat(i,N,1).*sqrt(small_sig2))/lambda;
end

X_d = [kron(diag(1:p),diag(sqrt(small_sig2)/lambda));
    zeros(N,K)];

%% Do Least Squares to get posterior
Y_star = [Y;Y_d]; X_star = [X; X_d];

%Get VAR coefficients
results.A = (X_star'*X_star)\(X_star'*Y_star);

%Get BVAR residuals and calcuate posterior covariance matrix
U = Y-X*results.A;
U_star = Y_star-X_star*results.A;
results.SIGMA = (U_star'*U_star)/(size(Y_star,1)-size(results.A,1));



end