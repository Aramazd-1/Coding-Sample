function [trendComponents, cyclicalComponents] = hpFilterFunction(y, dates, lambda)
    %HPFILTERFUNCTION Applies the HP filter to the given data and plots the results.
    %   [trendComponents, cyclicalComponents] = hpFilterFunction(y, dates, lambda)
    %   y - Data to be filtered
    %   dates - Corresponding dates for the data
    %   lambda - Smoothing parameter (e.g., 1600 for quarterly data)

    % Initialize components
    trendComponents = zeros(size(y));
    cyclicalComponents = zeros(size(y));

    % Apply HP filter
    for jj = 1:size(y,2)
        [trendComponents(:,jj), cyclicalComponents(:,jj)] = hpfilter(y(:,jj), lambda);
    end

    % Plotting
    figure;
    subplot(2,1,1);
    plot(dates, trendComponents(:,1), 'b-');
    title('Trend Component of EU GDP');
    xlabel('Year');
    ylabel('Value');

    subplot(2,1,2);
    plot(dates, cyclicalComponents(:,1), 'r-');
    title('Cyclical Component of EU GDP');
    xlabel('Year');
    ylabel('Value');
end

