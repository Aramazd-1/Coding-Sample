import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from describe_financial_data import describe_financial_data,ploty,apply_hill_estimator, fit_and_forecast_egarch,calculate_empirical_var, add_double_lines_to_latex, histogram_Normality_test, Correlogram, hill_estimator_both_tails,  garch_lag_selection, fit_and_forecast_garch
from termcolor import colored
from arch import arch_model
import numpy as np
from collections import Counter
from functools import partial
import statsmodels.api as sm

#%%
stock = '^STOXX50E'  #European top 50 stock index
# Download daily and weekly data
data_daily = yf.download(stock, interval='1d', start='1995-08-01', end = '2023-12-12')
data_weekly = yf.download(stock, interval='1wk')
print(data_daily.head())
print(data_weekly.head())
#%% Descriptive Statistics
describe_financial_data(data_daily,1) #NOTE that this function creates the 'Return' column needed in the computations below!
describe_financial_data(data_weekly,1,frequency='weekly')
print(colored("Description is over", 'yellow', 'on_grey' )) 

histogram_Normality_test(data_daily,column='Return',bins=100,frequency='daily')
histogram_Normality_test(data_weekly,column='Return',bins=100,frequency='weekly')
print(colored("Histogram and testing are over", 'yellow', 'on_grey' ))
#Ljung- Box test
lb_test = sm.stats.acorr_ljungbox(data_daily['Return'].dropna(), lags=35, return_df=True)
lb_test_weekly = sm.stats.acorr_ljungbox(data_weekly['Return'].dropna(), lags=35, return_df=True)
#Daily Correlograms
Correlogram(data_daily['Return'],-0.08,0.08 , titlez='Returns', frequency='daily') #Note that graphs scale is fixed here and not dynamic
Correlogram(data_daily['Return']**2,-0.3,0.3, titlez='Squared Returns',  frequency='daily')
Correlogram(np.cos(data_daily['Return']),-0.3,0.3, titlez= 'Cosine of Returns',  frequency='daily')

#Weekly correlograms
Correlogram(data_weekly['Return'],-0.15,0.15 , titlez='Returns', frequency='weekly') #Define this inside of a function so is less boring
Correlogram(data_weekly['Return']**2,-0.3,0.3, titlez='Squared Returns',  frequency='weekly')
Correlogram(np.cos(data_weekly['Return']),-0.3,0.3, titlez= 'Cosine of Returns',  frequency='weekly')
#%% Computing Hill estimator
significance_levels = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
hill_estimator_daily = pd.DataFrame(columns=['Left Tail Index', 'Right Tail Index'], index=significance_levels) 
hill_estimator_weekly = pd.DataFrame(columns=['Left Tail Index', 'Right Tail Index'], index=significance_levels)

vectorized_daily = apply_hill_estimator(data_daily['Return'], hill_estimator_daily) #Here a np.vectorize is implicitlu exploited to do the loop
vectorized_daily(significance_levels)

vectorized_weekly = apply_hill_estimator(data_weekly['Return'], hill_estimator_weekly) #Here a np.vectorize is implicitlu exploited to do the loop
vectorized_weekly(significance_levels)

#%% Garch lag selection
#We now move on to estimate stuff with percentages, this is done because the arch library need scale between 1 and 1000
#Note that here a loop will effectively be done over the various values, but it will be executed with a parallel pool
results_df_daily =  garch_lag_selection((data_daily['Return']*100).dropna().iloc[:int(len(data_daily['Return'])*2/3)],5,5) 
results_df_weekly =  garch_lag_selection((data_weekly['Return']*100).dropna().iloc[:int(len(data_weekly['Return'])*2/3)],5,5)
#%% Forecasting GARCH and VaR
#Note that data is appropriately turned into percentages and cleaned inside of the function below (basically the function replicates what is done above)
fit_daily, forecast_daily, VaR_daily,object_forecast_daily, Exceedence_daily = fit_and_forecast_garch(data_daily['Return'], 2/3, 1, 1, 0, mean='constant', power=2, vol= 'GARCH', dist= 'skewt',hor=5, frequency= 'daily')

fit_weekly, forecast_weekly,VaR_weekly, object_forecast_weekly, Exceedence_weekly = fit_and_forecast_garch(data_weekly['Return'], 2/3, 1, 1, 0, mean='constant', power=2, vol= 'GARCH', dist= 'skewt',hor=5, frequency= 'weekly')
#%%
ploty(forecast_daily['h.1'], title='Daily 1 step ahead volatility Forecast Garch', ylabel = 'Volatility')

ploty(forecast_weekly['h.5'], title='Weekly 1 step ahead volatility Forecast Garch', ylabel = 'Volatility')

ploty(forecast_daily['h.5'], title='Daily 5 step ahead volatility Forecast Garch', ylabel = 'Volatility')
#%% Forecasting EGARCH and VaR Daily and Weekly
fit_daily_e, forecast_daily_e, VaR_daily,object_forecast_daily_e, Exceedence_daily_e = fit_and_forecast_egarch(data_daily['Return'], 2/3, 1, 1, 1, mean='constant', dist= 'skewt',hor=5, frequency= 'daily')
fit_weekly_e, forecast_weekly_e, VaR_weekly,object_forecast_weekly_e, Exceedence_weekly_e = fit_and_forecast_egarch(data_weekly['Return'], 2/3, 1, 1, 1, mean='constant', dist= 'skewt',hor=5, frequency= 'weekly')
#%% Plotting some results and the residuals
ploty(forecast_daily_e['h.1'], title='Daily 1 step ahead volatility Forecast EGarch', ylabel = 'Volatility')

ploty(forecast_weekly_e['h.1'], title='Weekly 1 step ahead volatility Forecast EGarch', ylabel = 'Volatility')

ploty(forecast_daily_e['h.5'], title='Daily 5 step ahead volatility Forecast EGarch', ylabel = 'Volatility')

ploty(fit_daily.std_resid.dropna(), title='Daily Garch(1,1) residuals', ylabel = 'Residuals')

ploty(fit_weekly.std_resid.dropna(), title='Weekly Garch(1,1) residuals', ylabel = 'Residuals')

ploty(fit_daily_e.std_resid.dropna(), title='Daily EGarch(1,1) residuals', ylabel = 'Residuals')

ploty(fit_weekly_e.std_resid.dropna(), title='Weekly EGarch(1,1) residuals', ylabel = 'Residuals')

#%% Exporting tables to a txt file (NOTE: it will be saved in current working directory)
# Opens a new latex file and exports all the tables created by the code. It saves them in a txt, but by simply changing .txt to .tex a tex file would
# be returned to the user.
with open('tables_output.txt', 'w') as file:
    # Writing each LaTeX-formatted table to the file
    file.write(add_double_lines_to_latex(hill_estimator_daily.to_latex(index=True, escape=False, column_format='c c c')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(hill_estimator_weekly.to_latex(index=True, escape=False, column_format='c c c')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(lb_test.to_latex(index=True, escape=False, column_format='c c c')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(lb_test_weekly.to_latex(index=True, escape=False, column_format='c c c')))
    file.write('\n\n')

    
    file.write(add_double_lines_to_latex(results_df_daily.drop('Optimal', axis=1).to_latex(index=False, escape=False, column_format='|c c| c c c |')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(results_df_weekly.drop('Optimal', axis=1).to_latex(index=False, escape=False, column_format='|c c| c c c |')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(fit_daily.params.to_frame().T.to_latex(index=False, escape=False, column_format='c c c c c c')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(fit_weekly.params.to_frame().T.to_latex(index=False, escape=False, column_format='c c c c c c')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(fit_daily_e.params.to_frame().T.to_latex(index=False, escape=False, column_format='c c c c c c')))
    file.write('\n\n')
    
    file.write(add_double_lines_to_latex(fit_weekly_e.params.to_frame().T.to_latex(index=False, escape=False, column_format='c c c c c c')))
    file.write('\n\n')

    # Writing the Counters
    file.write(str(Counter(Exceedence_daily)))
    file.write('\n\n')
    
    file.write(str(Counter(Exceedence_weekly)))
    file.write('\n\n')
    
    file.write(str(Counter(Exceedence_daily_e)))
    file.write('\n\n')
    
    file.write(str(Counter(Exceedence_weekly_e)))
    file.write('\n\n')

# File is automatically closed after exiting the 'with' block


#%% Exporting tables to latex manually
# print(add_double_lines_to_latex(hill_estimator_daily.to_latex(index=True, escape=False,column_format='c c c')))
# print(add_double_lines_to_latex(hill_estimator_weekly.to_latex(index=True, escape=False,column_format='c c c')))
# print(add_double_lines_to_latex(lb_test.to_latex(index=True, escape=False,column_format='c c c')))
# print(add_double_lines_to_latex(lb_test_weekly.to_latex(index=True, escape=False,column_format='c c c')))

# print(add_double_lines_to_latex(results_df_daily.drop('Optimal', axis=1).to_latex(index=False, escape=False,column_format='|c c| c c c |') ) )
# print(add_double_lines_to_latex(results_df_weekly.drop('Optimal', axis=1).to_latex(index=False, escape=False,column_format='|c c| c c c |') ) )

# print( add_double_lines_to_latex( fit_daily.params.to_frame().T.to_latex(index=False, escape=False,column_format='c c c c c c') ) )
# print( add_double_lines_to_latex( fit_weekly.params.to_frame().T.to_latex(index=False, escape=False,column_format='c c c c c c') ) )

# print( add_double_lines_to_latex( fit_daily_e.params.to_frame().T.to_latex(index=False, escape=False,column_format='c c c c c c') ) )
# print( add_double_lines_to_latex( fit_weekly_e.params.to_frame().T.to_latex(index=False, escape=False,column_format='c c c c c c') ) )

# print(Counter(Exceedence_daily))
# print(Counter(Exceedence_weekly))

# print(Counter(Exceedence_daily_e))
# print(Counter(Exceedence_weekly_e))

