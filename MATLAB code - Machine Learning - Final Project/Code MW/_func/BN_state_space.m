function [BN_cycle,Info_decom,shock_decom,FEVD,Std_err] = BN_state_space(y,y_mean,p,VAR_parameters,target_variable,varagin)
% Benjamin Wong
% Monash University
% August 2021
% Calculates BN decomposition from a VAR using Morley (2002)
% as well as informational and shock decomposition (using Cholesky) as per Morley and Wong (2020)
%
%y                              vector(s) of variables
%y_mean                     the unconditional mean of the series in y
%p                              lags
%VAR Parameters      The estimated VAR parameters (to be used in conjuction
%                               with the structure of the various estimated BVARs)
%target_variable        The index of the variable to obtain cycle and decomposition
%
%vargin options     Decomposition - to do shock decomposition
%                   Standard_Errors - to calculate standard errors as per
%                   Kamber, Morley & Wong (2018)
% OUTPUT
%BN_cycle           Estimated BN cycle
%Info_decom         The info decomposition of the target variable
%shock_decom        The shock decomposition of the target variable
%FEVD               Variance decomposition - produced with shock
%                   decomposition
%Std_err            estimated standard error as per KMW(2018)

%%

A = VAR_parameters.A;
SIGMA = VAR_parameters.SIGMA;

[T N] = size(y);

%demean
y = y - repmat(y_mean,T,1);

%backcast data
y = [zeros(p,N);y];

%Create Data Matrices
Y = y(p+1:end,:);     %Cut Away first p lags (the backcasted stuff)
X = [];

for i = 1:p
    Z = y(p+1-i:end-i,:);
    X = [X Z];
end

%%

U = Y - X*A;
Companion = [A';eye(N*(p-1)) zeros(N*(p-1),N)];

%% BN Decomposition Starts here
%Add current period in to incoporate current information set to do BN decomposition
Z = X;
X = [X(2:end,:);
    Y(end,:) X(end,1:end-N)];

F = [A';
     eye(N*(p-1)) zeros(N*(p-1),N)];

H = [eye(N);zeros((p-1)*N,N)];        
        
bigeye = eye(N*p);

FinvIminusF = -F*((bigeye-F)\eye(N*p));
BN_cycle = FinvIminusF*X';
BN_cycle = BN_cycle(1:N,:)';

BN_cycle = BN_cycle(:,target_variable);

phi = -Companion*((bigeye-Companion)\eye(N*p));

%% Do decompositons of BN trend and cycle
% Setup inputs to calculating the informational and shock decomposition
bigeye_sec = [eye(N) zeros(N,N*(p-1))];
Info_decom = [];
shock_decom = [];
FEVD = [];
Std_err=[];
selector_vec = zeros(1,N);
selector_vec(target_variable) = 1;


if sum(strcmp(varagin,'Decomposition')) == 1
    %construct historical structural shocks
    
    eta = zeros(size(U));
    A0 = chol(SIGMA,'lower');
    A0_inv = A0\eye(N);
    
    for ii = 1:T
        eta(ii,:) = A0_inv*U(ii,:)';
        
    end
    
    Info_decom.cycle = zeros(size(X,1),N);
    Info_decom.trend = zeros(size(X,1),N);
    
    shock_decom.cycle = zeros(size(X,1),N);
    shock_decom.trend = zeros(size(X,1),N);
    
    % see formulaes in the paper
    for jj = 1:size(X,1)
        count = 0;
        
        for ii = jj:-1:1
            shock_vec = diag(U(ii,:));
            s_shocks_vec = diag(eta(ii,:));
            if ii == jj
                Info_decom.trend(jj,:) = selector_vec*bigeye_sec*((bigeye-Companion)\eye(N*p))*bigeye_sec'*shock_vec;
                shock_decom.trend(jj,:) = (selector_vec*bigeye_sec*((bigeye-Companion)\eye(N*p))*bigeye_sec'*A0*s_shocks_vec)';
            end
            
            Info_decom.cycle(jj,:)= Info_decom.cycle(jj,:)+ ...
                selector_vec*bigeye_sec*(phi*(Companion^count))*bigeye_sec'*shock_vec;
            
            shock_decom.cycle(jj,:)= shock_decom.cycle(jj,:)+ ...
                selector_vec*bigeye_sec*(phi*(Companion^count))*bigeye_sec'*A0*s_shocks_vec;
            
            count = count + 1;
        end
    end
    %% Calculate FEVD
    
    nFEVD = 101;    %number of FEVD periods to compute
    FEVD.cycle = zeros(N,nFEVD); %Group by variables
    
    for i = 1:nFEVD
        temp = bigeye_sec*phi*(Companion^(i-1))*bigeye_sec'*A0;
        temp = temp(target_variable,:).^2;
        if i == 1
            FEVD.cycle(:,1) = temp;
        else
            FEVD.cycle(:,i) = FEVD.cycle(:,i-1)+temp';
        end
    end
    %trend
    BigPhi = (bigeye-Companion)\eye(N*p);
    FEVD.trend = reshape(bigeye_sec*BigPhi*bigeye_sec'*A0,[],N);
    FEVD.trend = (FEVD.trend(target_variable,:).^2)';
    FEVD.trend = 100*FEVD.trend./repmat(sum(FEVD.trend),N,1);
    
end

%% Calculate Standard  Error

if sum(strcmp(varagin,'Standard_Errors')) == 1
    Q = zeros(size(Companion));
    Q(1:N,1:N) = VAR_parameters.SIGMA;
    
    % See Kamber Morley and Wong (REStat,2018, online appendx)
    SIGMA_X = (eye(size(Companion,1)^2)-kron(Companion,Companion))\Q(:);
    SIGMA_X = reshape(SIGMA_X,N*p,N*p);
    Std_err = phi*SIGMA_X*phi';
    Std_err = sqrt(diag(Std_err(1:N,1:N)));
    
    Std_err = Std_err(target_variable);
    
end


end

