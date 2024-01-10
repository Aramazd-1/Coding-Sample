function main
    rng('default'); %Setting up the random number generator for reproducibility

    M = 5000; %Number of Repetitions per Sample Combinations
    T_values = [25, 50, 100, 500, 1000]; %Vecotr of Sample dimensions
    h_values = [0.2, 0.8]; %Vector of h values
    P = 6; %Vector of lags

    for T = T_values %First loop over sample dimension
        for h = h_values %Second Loop over h values
            %Result matrix preparation
            results_AIC = zeros(P, M);
            results_BIC = zeros(P, M);
            results_HQC = zeros(P, M);

            Uncon_mean = [0.013 / (0.21 - 0.1 * h), (0.023 + 0.02 * h) / (0.21 - 0.1 * h)]; %The unconditional mean from point A will be first value in array

            for m = 1:M %Third loop over the Realizations
                data = generate_data(T, h, Uncon_mean); %DGP

                for p = 1:P
                    truncated_data = data(p+1:end, :);
                    [~, AIC, BIC, HQC] = estimate_VAR(truncated_data, p);

                    results_AIC(p, m) = AIC;
                    results_BIC(p, m) = BIC;
                    results_HQC(p, m) = HQC;
                end
            end

            indicator_AIC = results_AIC == min(results_AIC);
            indicator_BIC = results_BIC == min(results_BIC);
            indicator_HQC = results_HQC == min(results_HQC);

            sum_indicators_AIC = sum(indicator_AIC, 2);
            sum_indicators_BIC = sum(indicator_BIC, 2);
            sum_indicators_HQC = sum(indicator_HQC, 2);

            fprintf('T = %d, h = %.1f\n', T, h);
            disp('AIC');
            disp(sum(indicator_AIC, 2));
            disp('BIC');
            disp(sum(indicator_BIC, 2));
            disp('HQC');
            disp(sum(indicator_HQC, 2));
        end
    end
end

function data = generate_data(T, h, Uncon_mean)
    data = zeros(T, 2);

    for t = 2:T
        data(t, :) = Uncon_mean + h * (data(t-1, :) - Uncon_mean) + sqrt(1 - h^2) * randn(1, 2);
    end
end

function [VAR_model, AIC, BIC, HQC] = estimate_VAR(data, p)
    Y = data(p+1:end, :);
    X = [];
    for i = 1:p
        X = [X, lagmatrix(data, i)];
    end
    X = X(p+1:end, :);

    VAR_model = X \ Y;

    n = length(Y);
    k = numel(VAR_model);
    residuals = Y - X * VAR_model;
    sigma2 = sum(sum(residuals.^2)) / (n * 2);

    AIC = log(sigma2) + 2 * k / (n * 2);
    BIC = log(sigma2) + log(n) * k / (n * 2);
    HQC = log(sigma2) + 2 * log(log(n)) * k / (n * 2);
end