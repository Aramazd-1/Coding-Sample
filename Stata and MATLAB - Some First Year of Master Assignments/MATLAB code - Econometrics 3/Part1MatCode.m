 %% Clear workspace and set rng to default
clear
close all
clc
rng('default')
%% Call the Simulation function, returns table of result as latex output
Simulation; 
function Simulation
%Parameters definition
M = 5000; 
T = [25, 50, 100, 500, 1000]; 
h_values = [0.2, 0.8]; 
P = 6; 
lambda = [0.09, 0; 0, 0.04]; 
    %Storage matrices
results_AIC = zeros(P, M); %Basically this creates a  matrix full of zeros whose size is P x M 
results_BIC = zeros(P, M);
results_HQC = zeros(P, M);

%% Loop over parameters
    for T = T %First loop over Sample size
        for h = h_values %Second loop over h
            
            Uncon_mean = [0.013 / (0.21 - 0.1 * h), (0.023 + 0.02 * h) / (0.21 - 0.1 * h)]; 

            for m = 1:M %Third loop over samples, using parfor would increase speed but change results
                y = simulate_data(T, h, Uncon_mean,lambda); %Generate data

                for p = 1:P %Fourth loop over The lags
                    [~, AIC, BIC, HQC] = Estimation_Var(y, p,T,P); %Estimate model.

                    results_AIC(p, m) = AIC; %Criteria are stored for interaction m and lag p (each point of the inner loop)
                    results_BIC(p, m) = BIC; 
                    results_HQC(p, m) = HQC;
                end  %Stops loop over lags
            end %Stops loop over M
            
%% Indicator Function
            indicator_AIC = results_AIC == min(results_AIC,[],1); %Indicator function. Compares for each generated sample 6 fitted models and gives true to the best fit
            indicator_BIC = results_BIC == min(results_BIC,[],1); %Same dimension as results but is 1 or 0
            indicator_HQC = results_HQC == min(results_HQC,[],1);

            sum_indicators_AIC = sum(indicator_AIC, 2)/M; %Summation of the indicator function to compute the best. Divided by maximum, so between 0 and 1
            sum_indicators_BIC = sum(indicator_BIC, 2)/M; 
            sum_indicators_HQC = sum(indicator_HQC, 2)/M;

            critera_matrix = [sum_indicators_AIC, sum_indicators_BIC, sum_indicators_HQC];

            fprintf('T = %d, h = %.1f\n', T, h);
            disp(critera_matrix);

            %%  Export to LaTeX table
            filename = sprintf('table_T_%d_h_%.1f.tex', T, h);
            mat2latex(critera_matrix, filename, T, h);
        end
    end
end

function y = simulate_data(T, h, Uncon_mean, lambda)
%Initialize the data matrix at the unconditional mean
    y = repmat(Uncon_mean, T, 1); 
% Loop that generates the data at each point in time of the process
    for t = 3:T
        epsilon = mvnrnd([0, 0], lambda,1); %Error term generation
        A_1 = [0.5, 0.1; 0.4, 0.5]; %First matrix of coefficients to put in the DGP
        A_2 = [0, 0; h, 0]; %Second matrix of coefficients to put in the DGP
        A_0 = [0.02, 0.03];
        y(t, :) = A_0  + (A_1 * (y(t-1, :))')' + (A_2 * (y(t-2, :))')' + epsilon;
        %This generates the DGP gave by the question 1. 
    end
end

function [PI_hat, AIC, BIC, HQC] = Estimation_Var(y, p,T,P)  %Left is output right is input, Estimation function
    %% define dependent variable W and regressors X
    W = y(P+1:T,:);     X = y(6:T-1,:);
    X = [X ones(T-P,1)]; % adds the constant
    for i=1:p-1 %loop generating regressors
        X = [X,y(P-i:T-i-1,:)];
    end
    %% Interpreting the hint we decided to use P in place of p and T in place of t. Hopefully this doesn't introduce efficiency issues due to less usage of data.
    %T_corr = T-p; Also here one could use this to compute the Criterion,
    %found different results online. After testing with both no qualitative
    %difference is shown, results slightly differ.

    % estimate parameter matrix PI, errors E and the v-cov matrix Lambda 
    PI_hat = (X\W);
    E_hat = W - X * PI_hat;
    Lambda_hat = E_hat'*E_hat/(T - size(X, 2));

    AIC = log(det(Lambda_hat)) + (2*p*4)/T;
    BIC = log(det(Lambda_hat)) + (log(T)*p*4)/T;
    HQC = log(det(Lambda_hat)) + (2*log(log(T))*p*4)/T;

end %Close estimation function definition


function mat2latex(matrix, filename, T, h) %This function prints out the needed tables as latex output

    fid = fopen(filename, 'w'); %This allows matlab to write into the file that will become the latex table
    [rows, cols] = size(matrix); %size of matrix

   fprintf(fid, '\\begin{table}[ht]\n');
    fprintf(fid, '\\captionsetup{justification=raggedright,singlelinecheck=false}\n');
    fprintf(fid, '\\caption*{T = %d, h = %.1f}\n', T, h);
    fprintf(fid, '\\begin{tabular}{c|%s}\n', repmat('c', 1, cols)); %Prints table header
    fprintf(fid, '  $p$ & AIC & BIC & HQC \\\\\\hline\n'); %column names+ align


    for i = 1:rows  %row loop
        fprintf(fid, '  %d', i); %writes the index of the row
        for j = 1:cols %column loop
            fprintf(fid, ' & %.4f', matrix(i, j));  %This prints the actual element for each column
        end %end col loop
        fprintf(fid, ' \\\\\n');
    end %end row loop

    fprintf(fid, '\\end{tabular}\n'); 
    fprintf(fid, '\\end{table}\n'); %Table end
    fclose(fid);
end %close latex function


