% covid corrected
% BVAR_parameters.natural_conjugate =...
%     BVAR_Minnesota_COVID_MLE(y,p,lambda,Tcovid,nCovidParam);
BVAR_parameters.natural_conjugate =...
    BVAR_Minnesota_COVID_MLE(y,p,lambda,Tcovid,nCovidParam);

VAR_parameters.A = BVAR_parameters.natural_conjugate.A;
VAR_parameters.SIGMA = BVAR_parameters.natural_conjugate.SIGMA;

[Gap_no_block_exo,~,~,~,~] = ...
    BN_state_space(y,mean(y),p,VAR_parameters,target_var,[]);