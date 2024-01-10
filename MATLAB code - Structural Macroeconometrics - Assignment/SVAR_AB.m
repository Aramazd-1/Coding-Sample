%% SVAR
clear 
clc
close all
clear global     % cleans all gloabl variables
addpath([pwd '\Functions']); % This adds Functions to the available folders to extract functions for the code
%This approach allows modularity, pwd assumes Functions is a subfolder of
%the present working directory
%%
global p              % # of VAR lags
global T              % # of observations in dataset   
global ParamNumberA   % needs to select the structural parameters to estimate in the structural form for matrix A
global ParamNumberB   % needs to select the structural parameters to estimate in the structural form for matrix B
%% dataset Loading if xlsx is used
DataOriginal = readtable("DATI_FISCAL.xlsx"); %This is the data matrix with also the dates and column names;
VariablesSelected = [4,3,2]; %Will be the vector of variables used out of the dataset (Tax(4), G(3), GDP(2))
EndSample = 228 ; % Specifies which is the last observation. Useful if sub-samples need to be chosen. Currently is just "size(DataOriginal,1)"
Data = table2array(DataOriginal(:,VariablesSelected)); %This matrix contains only the values, not labels nor dates

%% Specification 
p = 4;                  % # of lags of estiated VAR
T = size(Data,1)-p;   
M = size(Data,2);      % VAR dimension M  
W = Data(p+1:EndSample,:);   % corresponds to W matrix of data in the slides
%% VAR Estimate (reduced form)
VAR_Const = NaN(M, 1);        % VAR constant, Mx1 vector; note that
VAR_Trend = NaN(M, 1);        % if you put zero in place of NaN, that will be managed as a zero constraint in estimation
%dummyVar  = table2array(DataOriginal(:,5));                      
VAR_Pi=cell(1, p);
for i = 1:p
    VAR_Pi{i} = NaN(M, M);  % Each matrix is MxM with NaN values
end
 
        
VAR = varm('Constant',VAR_Const,'AR',VAR_Pi, 'Trend', VAR_Trend); % Definining the VAR object to be estimate afterwards 
[EstVAR,EstSE,logLikVAR,Residuals] = estimate(VAR,Data);  % Estimation by ML, the "estimate()" command is provided by Matlab

Const = EstVAR.Constant;     % Here we put the estimates in the left-hand-side matrices            
Trend = EstVAR.Trend; 
mP=cell(1,p);                %Initializing estimated Pis

for i =1:p
    mP{i} = EstVAR.AR{1,i}; %This loop extracts the estimates of the p Pi matrices
end

% Covariance matrix
Sigma_u = (Residuals'*Residuals)/T;  % estimated covariance matrix could also be (U'U)/T
Sigma_u_sample = Sigma_u;
%% SVAR estimation (AB-model) with ML
          
ParamNumberB =[1  ...     % first column of B, notice all but first element is set to 0
              4 5 ...     % second column of B, notice that position (1,2) corresponding to 4 is set to zero
              9]';        % third column of B, positions (1,3) and (2,3) corresponding to 7 and 8 are set to zero 
ParamNumberA = [ 3  ...
                6   ...
                ]';         
% size(ParamNumber,1)   
Matrix_Selected = zeros(size(Sigma_u_sample,1),size(Sigma_u_sample,1));

for c_par = 1 : size(ParamNumberB,1)
Matrix_Selected(ParamNumberB(c_par,1)) = 1;    % this loop puts parameters for B in matrix form
end

for c_par = 1 : size(ParamNumberA,1)
Matrix_Selected(ParamNumberA(c_par,1)) = 1;    % this loop puts parameters for A in matrix form
end
StructuralParamB = size(ParamNumberB,1);        % dimesion of vector of structural parameters (beta in the slides)
StructuralParamA = size(ParamNumberA,1);        % dimesion of vector of structural parameters (beta in the slides)
InitialValue_B_A = (randn(StructuralParamA+StructuralParamB,1)/1000)';  % initial random values for B in order to enter the likelihood maximization
  
%% Likelihood Maximization
options = optimset('MaxFunEvals',200000,'TolFun',1e-200,'MaxIter',200000,'TolX',1e-200); 
% Code below creates an anonymous function (a wrapper). This way we avoid declaring a global prameter
% but multi argument can be passed indirectly to fminunc. The function
% takes sigma_u as fixed and is passed the "teta" argument in fminunc.
objectiveFunction = @(teta) Likelihood_SVAR_AB(teta, Sigma_u);
[StructuralParam_Estimation_MATRIX,Likelihood_SVAR,exitflag,output,grad,Hessian_MATRIX] = fminunc(objectiveFunction, InitialValue_B_A, options);
%%
SE_Hessian_MATRIX = diag(inv(Hessian_MATRIX)).^0.5;  % computes Hessian-based standard errors
A = [1, 0, -2.08;
    0, 1, 0;
    NaN, NaN, 1];
B = zeros(size(Sigma_u,1),size(Sigma_u,1));
SE_B = zeros(size(Sigma_u,1),size(Sigma_u,1));
SE_A = zeros(size(Sigma_u,1),size(Sigma_u,1));
HSelectionB = zeros(M*M,StructuralParamB);       % this matrix corresponds to selection matrix S_B in slides 
HSelectionA = zeros(M*M,StructuralParamA);

for c_par = 1 : size(ParamNumberB,1)
B(ParamNumberB(c_par,1)) = StructuralParam_Estimation_MATRIX(c_par);     % puts the estimated elements of B in the right place  
SE_B(ParamNumberB(c_par,1))= SE_Hessian_MATRIX(c_par);
HSelectionB(ParamNumberB(c_par,1),c_par) = 1;                            % puts "1" in the correct place of selection matrix S_B 
end
for c_par = 1 : size(ParamNumberA,1)
A(ParamNumberA(c_par,1)) = StructuralParam_Estimation_MATRIX(c_par+size(ParamNumberB,1));     % puts the estimated elements of B in the right place  
SE_A(ParamNumberA(c_par,1))= SE_Hessian_MATRIX(c_par+size(ParamNumberB,1));
HSelectionA(ParamNumberA(c_par,1),c_par) = 1;                            % puts "1" in the correct place of selection matrix S_B 
end
%%
% Sign normalization  (recall that identification holds up to sign normalization)  
A = DiagonalSignNorm(A); %This function performs sign normalization and is stored in functions folder
B = DiagonalSignNorm(B); %Basically, every diagonal element is turned positive if not already positive
Likelihood_SVAR = -1*Likelihood_SVAR ;  % returns the log-likelihood of SVAR
LR_test_overid = -2*(Likelihood_SVAR - logLikVAR);


df = M*(M+1)/2 -(size(ParamNumberA,1)+size(ParamNumberB,1));
if (df>0) 
    PVal = 1-chi2cdf(LR_test_overid,df);
end    

disp(B),  disp(A) ,disp(SE_B) ,disp(SE_A), disp(Likelihood_SVAR), disp(LR_test_overid)
%% Dynamic Multipliers
C_IRF = A^(-1)*B;   % instantaneous impact at h=0
HorizonIRF = 20; 
CompanionMatrix = [mP{1} mP{2} mP{3} mP{4};eye(M*(p-1)) zeros(M*(p-1),M)];           % VAR companion matrix
J=[eye(M) zeros(M,M*(p-1))];                          % selection matrix J used in IRF computation                        
    TETA = zeros(M,M,HorizonIRF+1); %Preallocates for speed
    for h = 0 : HorizonIRF
    TETA(:,:,h+1)=J*CompanionMatrix^(h)*J'*C_IRF;
    end

Impulse_G_G= C_IRF(2,2);   %First variable is originating the shock(2)/G, second variable who receives impact(2)/G, evaluated at h=0
Impulse_tax_tax = C_IRF(1,1);
GDP_over_tax =mean( ( exp(Data(:,3)) ./ exp(Data(:,1)) ) ) ; %Ratios needed to compute multipliers
GDP_over_G = mean( exp(Data(:,3))./exp(Data(:,2)) ) ;
Dynamic_tax_multiplier=zeros(1,HorizonIRF+1);
Dynamic_Spending_multiplier=zeros(1,HorizonIRF+1);
for h = 0: HorizonIRF
    Impulse = J*CompanionMatrix^(h)*J'*A^(-1)*B; %Usual matrix of impulse responses
    Impulse_GDP_tax = Impulse(3,1) ;    %Computes value used below
    Impulse_GDP_G   = Impulse (3,2) ;   %Computes value used below
    Dynamic_tax_multiplier(1,h+1) = (Impulse_GDP_tax/Impulse_tax_tax)*GDP_over_tax;
    Dynamic_Spending_multiplier(1,h+1) = (Impulse_GDP_G /Impulse_G_G)*GDP_over_G;
end
%% ********************** BOOTSTRAP **********************
%WARNING: the original approach could get unstable due to where minimization is inizialized, sometimes the algorithm crashes. Usually just re-running it
%solves the issue.

%
Boot_Init= [B(ParamNumberB(1,1)),B(ParamNumberB(2,1)),B(ParamNumberB(3,1)),B(ParamNumberB(4,1)),A(ParamNumberA(1,1)),A(ParamNumberA(2,1))];
BootstrapIterations = 1000; 
quant = [5,95]; % quantile bootstrap to build 90% confidence intervals
%parfor could be used in more advanced / computationally intesnsive implementations to ensure computation is run in parallel. However, a new
%loop construction approach would be needed.
 Dynamic_tax_multiplier_Boot= zeros(1,HorizonIRF+1,BootstrapIterations); %Pre-Allocation of variables used in the loop for higher computational speed
 Dynamic_Spending_multiplier_Boot= zeros(1,HorizonIRF+1,BootstrapIterations); 
 TETA_Boot = zeros(M,M,HorizonIRF+1,BootstrapIterations) ;
tic
 for boot = 1 : BootstrapIterations

    %  **** iid bootstrap ****
    
    TBoot=datasample(1:T,T); % resample from 1 to T
    Residuals_Boot=Residuals(TBoot,:);  % bootstrap errors 

    DataSet_Bootstrap=zeros(T+p,M);
    DataSet_Bootstrap(1:p,:)=Data(1:p,:); % set the first p elements equal to the original sample values
 
        for t = 1+p : T+p
        DataSet_Bootstrap(t,:)=Const + Trend + mP{1} * DataSet_Bootstrap(t-1,:)' +...
                                       mP{2} * DataSet_Bootstrap(t-2,:)' + ...
                                       mP{3} * DataSet_Bootstrap(t-3,:)' + ...
                                       mP{4} * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-p,:)';
        end

    DataSet_Bootstrap=DataSet_Bootstrap(end-T+1:end,:);
    
    [EstVAR_Boot,EstSE_Boot,logLikVAR_Boot,Residuals_Boot] = estimate(VAR,DataSet_Bootstrap); 
    mP1_Boot = EstVAR_Boot.AR{1,1};
    mP2_Boot = EstVAR_Boot.AR{1,2};
    mP3_Boot = EstVAR_Boot.AR{1,3};
    mP4_Boot = EstVAR_Boot.AR{1,4};
    Sigma_u_Boot = (Residuals_Boot'*Residuals_Boot)/T;
    %Note that below the warnings are set to off so the "local minimum found" alert is
    %not spammed
    objectiveFunction2 = @(teta) Likelihood_SVAR_AB(teta, Sigma_u_Boot); %Same anonymous function
    options = optimset('MaxFunEvals',200000,'TolFun',1e-500,'MaxIter',200000,'TolX', 1e-50, 'Display', 'off');   
    [StructuralParam_Estimation_Boot,Likelihood_SVAR,exitflag,output,grad,Hessian_MATRIX] = fminunc(objectiveFunction2,Boot_Init', options);
    A_Boot = [1, 0, -2.08;
    0, 1, 0;
    NaN, NaN, 1];
    B_Boot = zeros(size(Sigma_u_Boot,1),size(Sigma_u_Boot,1));
    
    for c_par = 1 : size(ParamNumberB,1)
    B_Boot(ParamNumberB(c_par,1)) = StructuralParam_Estimation_Boot(c_par);     % puts the estimated elements of B in the right place                    % puts "1" in the correct place of selection matrix S_B 
    end
    for c_par = 1 : size(ParamNumberA,1)
    A_Boot(ParamNumberA(c_par,1)) = StructuralParam_Estimation_Boot(c_par+size(ParamNumberB,1));     % puts the estimated elements of B in the right place                    % puts "1" in the correct place of selection matrix S_B 
    end
    
    A_Boot = DiagonalSignNorm(A_Boot); %This function performs sign normalization and is stored in functions folder
    B_Boot = DiagonalSignNorm(B_Boot); %Basically, every diagonal element is turned positive if not already positive
    
    J=[eye(M) zeros(M,M*(p-1))]; 
    CompanionMatrix_Boot = [mP1_Boot mP2_Boot mP3_Boot mP4_Boot;
                            eye(M*(p-1)) zeros(M*(p-1),M)];
    C_IRF_Boot = A_Boot^(-1)*B_Boot;
    Impulse_tax_tax_Boot=C_IRF_Boot(1,1);
    Impulse_G_G_Boot=C_IRF_Boot(2,2);
    for h = 0 : HorizonIRF
    TETA_Boot(:,:,h+1,boot)=J*CompanionMatrix_Boot^(h)*J'*A_Boot^(-1)*B_Boot;
    end    


    for h = 0: HorizonIRF
    Impulse_Boot = J*CompanionMatrix_Boot^(h)*J'*A_Boot^(-1)*B_Boot; %Usual matrix of impulse responses
    Impulse_GDP_tax_Boot = Impulse_Boot(3,1) ; %Computes value used below
    Impulse_GDP_G_Boot   = Impulse_Boot (3,2) ;%Computes value used below
    Dynamic_tax_multiplier_Boot(1,h+1,boot) = (Impulse_GDP_tax_Boot/Impulse_tax_tax_Boot)*GDP_over_tax;
    Dynamic_Spending_multiplier_Boot (1,h+1,boot) = (Impulse_GDP_G_Boot /Impulse_G_G_Boot)*GDP_over_G;
    end
end   
toc % Usually takes around 24 seconds to run with 1000 iterations
disp('Bootstrap Completed')
IRF_Inf_Boot = prctile(TETA_Boot,quant(1),4); %Extracts extrema of confidence bands
IRF_Sup_Boot = prctile(TETA_Boot,quant(2),4);

IRF_Inf_tax_Boot = prctile(Dynamic_tax_multiplier_Boot,quant(1),3);
IRF_Sup_tax_Boot = prctile(Dynamic_tax_multiplier_Boot,quant(2),3);

IRF_Inf_Spending_Boot = prctile(Dynamic_Spending_multiplier_Boot,quant(1),3);
IRF_Sup_Spending_Boot = prctile(Dynamic_Spending_multiplier_Boot,quant(2),3);
%% IRF
LineWidth_IRF = 1.5;
FontSizeIRFGraph = 14;

Titles=cell(1,3);
Titles{1,1}='$$\varepsilon_{TAX}$$ $$Shock$$';
Titles{1,2}='$$\varepsilon_{G}$$ $$Shock$$';
Titles{1,3}='$$\varepsilon_{GDP}$$ $$Shock$$';


YLabel=cell(3,1);
YLabel{1,1}='$$TAX$$';
YLabel{2,1}='$$G$$';
YLabel{3,1}='$$GDP$$';
index = 1;
Shock_1=[1 1 1];

figure(1)
for jr = 1 : M
    for jc = 1 : M
    TETA_Iter_Sample = squeeze(TETA(jr,jc,:));
    TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot(jr,jc,:));
    TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot(jr,jc,:));
    subplot(M,M,index)  
    x = 1:1:HorizonIRF+1;
    y= TETA_Iter_Sample'*Shock_1(jr);
    plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
    hold all
    plot(TETA_Iter_Boot_Inf,'--r', 'LineWidth',LineWidth_IRF);
    plot(TETA_Iter_Boot_Sup,'--r', 'LineWidth',LineWidth_IRF);
    plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);
    ylabel(YLabel{jr,1},'interpreter','latex');
    title(Titles{1,jc},'interpreter','latex');
    set(gca,'FontSize',FontSizeIRFGraph);      
    axis tight
    index=index+1;
    
    end
end

figure(2)
 TETA_1 = squeeze(Dynamic_tax_multiplier(1,:));
 TETA_Iter_Boot_Inf_1 = squeeze(IRF_Inf_tax_Boot(1,:));
 TETA_Iter_Boot_Sup_1 = squeeze(IRF_Sup_tax_Boot(1,:));    
 x = 1:1:HorizonIRF+1;
 y= TETA_1'*Shock_1(1);
 plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
 hold all
 plot(TETA_Iter_Boot_Inf_1,'--r', 'LineWidth',LineWidth_IRF);
 plot(TETA_Iter_Boot_Sup_1,'--r', 'LineWidth',LineWidth_IRF);
 plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);     
 ylabel(YLabel{3,1},'interpreter','latex');
 title('$$Dynamic Multiplier_{Tax}$$','interpreter','latex');
 set(gca,'FontSize',FontSizeIRFGraph);
 axis tight
 
figure(3)
 TETA_2 = squeeze(Dynamic_Spending_multiplier(1,:));
 TETA_Iter_Boot_Inf_2 = squeeze(IRF_Inf_Spending_Boot(1,:));
 TETA_Iter_Boot_Sup_2 = squeeze(IRF_Sup_Spending_Boot(1,:));    
 x = 1:1:HorizonIRF+1;
 y= TETA_2'*Shock_1(1);
 plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
 hold all
 plot(TETA_Iter_Boot_Inf_2,'--r', 'LineWidth',LineWidth_IRF);
 plot(TETA_Iter_Boot_Sup_2,'--r', 'LineWidth',LineWidth_IRF);
 plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);     
 ylabel(YLabel{3,1},'interpreter','latex');
 title('$$Dynamic Multiplier_{G}$$','interpreter','latex');
 set(gca,'FontSize',FontSizeIRFGraph);
 axis tight
  
    


    