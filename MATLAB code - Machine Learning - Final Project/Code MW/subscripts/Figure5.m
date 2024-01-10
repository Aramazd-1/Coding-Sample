%% Univariate

univariate_BN_cycle = multivariate_BN(y(:,3),p);

%% DROP variables

%drop 4 variables
y_alt1 = y;
y_alt1(:,[4 12 13 16]) = [];
Series_alt1 = Series;
Series_alt1([4 12 13 16])=[];

BVAR_parameters =...
    BVAR_block_exogenous(y_alt1,p,lambda,nBlockExo,nburn,total_draws);

VAR_parameters.A = mean(BVAR_parameters.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);

[Gap_alt1,~,~,~,~] = ...
    BN_state_space(y_alt1,mean(y_alt1),p,VAR_parameters,target_var,'Decomposition');

disp('12 variable estimation done')
toc

%% Drop term spread

y_alt2 = y;
y_alt2(:,10) = [];
Series_alt2 = Series;
Series_alt2(10)=[];

BVAR_parameters =...
    BVAR_block_exogenous(y_alt2,p,lambda,nBlockExo,nburn,total_draws);

VAR_parameters.A = mean(BVAR_parameters.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);


[Gap_alt2,~,~,~,~] = ...
    BN_state_space(y_alt2,mean(y_alt2),p,VAR_parameters,target_var,'Decomposition');

disp('Drop Term spread estimation done')
toc

%% Drop RER

y_alt3 = y;
y_alt3(:,16) = [];
Series_alt3 = Series;
Series_alt3(16)=[];

BVAR_parameters.covid_correction =...
    BVAR_block_exogenous(y_alt3,p,lambda,nBlockExo,nburn,total_draws);

VAR_parameters.A = mean(BVAR_parameters.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);

[Gap_alt3,~,~,~,~] = ...
    BN_state_space(y_alt3,mean(y_alt3),p,VAR_parameters,target_var,'Decomposition');

disp('Drop RER estimation done')
toc