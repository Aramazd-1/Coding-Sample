BVAR_parameters.KW2019Q4 =...
    BVAR_block_exogenous(y(1:Tcovid-1,:),p,0.2,nBlockExo,nburn,total_draws);

VAR_parameters.A = mean(BVAR_parameters.KW2019Q4.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.KW2019Q4.SIGMA_post,3);

[Gap_compare_other_covid_adjustment(:,1),~,~,~,~] = ...
    BN_state_space(y,mean(y(1:Tcovid-1,:)),p,VAR_parameters,target_var,[]);

disp('Block Exogeneous Without Covid sample done')
toc
%%
BVAR_parameters.KW_full_sample =...
    BVAR_block_exogenous(y,p,lambda,nBlockExo,nburn,total_draws);


VAR_parameters.A = mean(BVAR_parameters.KW_full_sample.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.KW_full_sample.SIGMA_post,3);

[Gap_compare_other_covid_adjustment(:,2),~,~,~,~] = ...
    BN_state_space(y,mean(y),p,VAR_parameters,target_var,[]);

disp('Block Exogeneous Without Covid correction done')
toc

%% MH (fully bayesian approach)

%number of burn-in for estimating s_t
MH_nburn = 20000;
%number of posterior draws for estimating s_t. Note that there is a much
%larger number of draws here
MH_n_posterior = 50000;

mcmc =    BVAR_block_exogenous_covid_adjustment(y,p,lambda,nBlockExo,Tcovid,nCovidParam,MH_nburn,MH_n_posterior);


BVAR_parameters.MH_Correction.SIGMA = mean(mcmc.sigma,3);

BVAR_parameters.MH_Correction.beta= mean(mcmc.beta,2);

BVAR_parameters.MH_Correction.A = zeros(N*p,N);
Dom_block = reshape(BVAR_parameters.MH_Correction.beta(nBlockExo*nBlockExo*p+1:end),N*p,[]);

BVAR_parameters.MH_Correction.A(:,nBlockExo+1:end) = Dom_block;

for jj = 1:nBlockExo
    for ii = 1:p
        for kk = 1:nBlockExo
            BVAR_parameters.MH_Correction.A((ii-1)*N + kk,jj) ...
                = BVAR_parameters.MH_Correction.beta((jj-1)*(nBlockExo*p) + ((ii-1)*nBlockExo)+kk);
        end
    end
end

[Gap_MH_Correction,~,~,~,~] = ...
    BN_state_space(y,mean(y),p,BVAR_parameters.MH_Correction,target_var,'Decomposition');

disp('Fully bayesian estimation done')
toc
