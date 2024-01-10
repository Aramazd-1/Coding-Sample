%% Replicates Morley, Rodriguez Palenzuela, Sun and Wong
% Estimating the Euro Area Output Gap Using Multivariate Information and Addressing the COVID-19 Pandemic⋆
%%
clear
clc

addpath(genpath('_func'));
addpath(genpath('file_for_COVID_adjustment'));
addpath(genpath('figures'));
addpath(genpath('subscripts'));

%% Choose Figures to replicate. Choose 1 to replicate. 0 to not replicate
options.Figure1 = 1;
options.Figure2 = 0; 
options.Figure3 = 1; 
options.Figure4 = 1;
options.Figure5 = 1;
options.Figure6 = 1;
options.Figure7 = 1;
options.Figure8 = 0; 


%% Make dataset

%load non FRED data
rawdata.nonFRED = xlsread('dataset.xlsx','non_FRED','B3:M95');
%load the transformation for the non FRED data
rawdata.tcode = xlsread('dataset.xlsx','non_FRED','B2:M2');

for jj = 1:size(rawdata.nonFRED,2)
    if rawdata.tcode(jj) == 0 %level
        y(:,jj) = rawdata.nonFRED(2:end,jj);
    end

    if rawdata.tcode(jj) == 1 % log level
        y(:,jj) = 100*log(rawdata.nonFRED(2:end,jj));
    end

    if rawdata.tcode(jj) == 2 % log difference
        y(:,jj) = 100*diff(log(rawdata.nonFRED(:,jj)));
    end

    if rawdata.tcode(jj) == 3 % difference
        y(:,jj) = diff(rawdata.nonFRED(:,jj));
    end
end

%load FRED data
rawdata.FRED = xlsread('dataset.xlsx','FRED','B7:F99');

%add in the FRED data where RER is last, US GDP and oil price first
y = [100*diff(log(rawdata.FRED(:,1))) 100*log(rawdata.FRED(2:end,5)) y 100*log(rawdata.FRED(2:end,4))];

dates = (1999:0.25:2019.75)'; %dates label  'US GDP','Real Oil Price',

Series = {'US GDP','Real Oil Price','euro area GDP','IP','Employment','Housing Permits','CPI',...
    'Policy Rate','Hours Worked','Term Spread','CAPU','Unemployment','PMI','Risk Spread', ...
    'RER'...
};

%% PRELIMINARIES

lambda = 0.2;       % Shrinkage hyperparameter. Suggested value by the literature (when covid is not considered. (Carriero et al., 2015)
nBlockExo = 2;                              
target_var = nBlockExo+1;           
[T,N] = size(y);
total_draws = 15000;
nburn = 7000 ;
p=4 ; 

tic

%% Estimate baseline

BVAR_parameters =...
    BVAR_block_exogenous(y,p,lambda,nBlockExo,nburn,total_draws);

VAR_parameters.A = mean(BVAR_parameters.A_post,3);
VAR_parameters.SIGMA = mean(BVAR_parameters.SIGMA_post,3);

[Baseline,Info_decom_baseline,shock_decom_baseline,~,~] = ...
    BN_state_space(y,mean(y),p,VAR_parameters,target_var,'Decomposition');

disp('Baseline done')
toc
%% ESTIMATE OTHER MODELS
if options.Figure2 == 1
    disp('Figure 2 estimations started')
    Figure2
    disp('Figure 2 estimations done')
end

if options.Figure3 == 1 || options.Figure6 == 1
    disp('Figure 3 and/or 6 estimations started')
    Figure3_6
    disp('Figure 3 and/or 6 estimations done')
end

if options.Figure5 == 1
    disp('Figure 5 estimations started')
    Figure5
    disp('Figure 5 estimations done')
end

if options.Figure8 == 1
    disp('Figure 8 estimations started')
    Figure8
    disp('Figure 8 estimations done')
end

plotting_script

disp('Total time taken')
toc
disp('====Completed=====')
