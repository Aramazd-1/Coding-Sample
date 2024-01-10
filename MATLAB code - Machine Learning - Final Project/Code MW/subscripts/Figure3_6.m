BVAR_parameters =...
    BVAR_block_exogenous(y,p,0.75,nBlockExo,nburn,total_draws);

VAR_parameters.A = mean(BVAR_parameters.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);

[Gap_point2,Info_decom_point2,~,~,~] = ...
    BN_state_space(y,mean(y),p,VAR_parameters,target_var,'Decomposition');

disp('\lambda = 0.75 estimation done')
toc

if options.Figure6 == 1
    Info_decom_other_baseline = sum(Info_decom_baseline.cycle,2) - Info_decom_baseline.cycle(:,9);
    Info_decom_other_point2 = sum(Info_decom_point2.cycle,2) - Info_decom_point2.cycle(:,9);
end
%%

if options.Figure3 == 1
    BVAR_parameters =...
        BVAR_block_exogenous(y,p,0.075,nBlockExo,nburn,total_draws);

    VAR_parameters.A = mean(BVAR_parameters.A_post,3);
    VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);

    [Gap_point075,~,~,~,~] = ...
        BN_state_space(y,mean(y),p,VAR_parameters,target_var,'Decomposition');

    disp('\lambda = 0.075 estimation done')
    toc

    BVAR_parameters =...
        BVAR_block_exogenous(y,p,0.9,nBlockExo,nburn,total_draws);

    VAR_parameters.A = mean(BVAR_parameters.A_post,3);
    VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);

    [Gap_point9,~,~,~,~] = ...
        BN_state_space(y,mean(y),p,VAR_parameters,target_var,'Decomposition');

    disp('\lambda = 0.9 estimation done')
    toc
end