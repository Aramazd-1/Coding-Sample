import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from termcolor import colored
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf as acf
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed

def describe_financial_data(data, logReturn=False, frequency ='daily'):
    """
    It will take data as input and perform:
    -Computation of returns
    -Descriptive Statistics
    -Time series plot
    -ADF test
    -Volatility analysis
    """
    # Calculating Returns based on the type
    if logReturn == 0:
        data['Return'] = data['Adj Close'].pct_change() #Normal PCT change
    else:
        data['Return']= np.log(data['Adj Close'])-np.log(data['Adj Close'].shift(1))#Log variation (approximately a percentage change)
    # Descriptive Statistics
    print("Descriptive Statistics for Returns:")
    print(data['Return'].describe())
    print(f"Skewness: {data['Return'].skew()}, Excess Kurtosis: {data['Return'].kurtosis()}") #Means that 3 is subtracted from Kurtosis
    #Optional plot style 
    # plt.style.use("seaborn-dark")
    # for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    #     plt.rcParams[param] = '#212946'  # bluish dark grey

    # for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    #     plt.rcParams[param] = '0.9'  # very light grey
    # plt.rcParams['grid.color'] = '#2A3459'
    # Time Series Plots 
    #data['Adj Close'].dropna().plot(title=f'{frequency.capitalize()} Adjusted Close Price')
    #plt.grid(True, color='#2A3459', alpha=0.75)
    #plt.savefig(f'D:\Study\Python\Financial Time Series\{frequency.capitalize()} Adjusted Close Price')
    #plt.show()
    
    data['LAdj Close']=np.log(data['Adj Close'])
    data['LAdj Close'].dropna().plot(title=f'{frequency.capitalize()} Log Adjusted Close Price')
    #plt.grid(True, color='#2A3459', alpha=0.75)
    plt.show() 
    
  
    data['Return'].dropna().plot(title=f'{frequency.capitalize()} Returns')
    #plt.grid(True, color='#2A3459', alpha=0.75)
    plt.show()
    
    # Augmented Dickey-Fuller Test
    test_takers=['Adj Close','LAdj Close','Return'] #List of variable to be tested with ADF
    adf_results={}
    for taker in test_takers:
            adf_result = adfuller(data[taker].dropna())
            adf_results[taker] = {'ADF Statistic': adf_result[0], 'p-value': adf_result[1]}
            print(f'ADFstatistics for {taker}: {adf_result[0]}')
            print(f'p-value for {taker}:{adf_result[1]}')
            if adf_result[1] > 0.05:
                print(colored(f"ADF test for {taker} detects a unit roott, indicating non-stationarity.", 'red','on_grey'))
            else:
                print(colored(f"ADF test for {taker} does not detect a unit root, indicating stationarity", 'green','on_grey'))
    # Volatility Analysis (30-day Rolling Standard Deviation)
    if frequency == 'weekly':
        rolling_window = 4
        name= '4-Weeks'
    else:
        rolling_window = 30
        name= '30-Day'
    data['Rolling_Std'] = data['Return'].rolling(window=rolling_window).std()
    data['Rolling_Std'].dropna().plot(title=f'{name} Rolling Standard Deviation')
    plt.show()
    


    return {taker: results['p-value'] for taker, results in adf_results.items()}, data['Return']


def histogram_Normality_test(data, column='Return', bins=50, frequency='daily'):
        """
        Plots a histogram of the specified column from the DataFrame. Moreover, it makes Normality tests on it.
    
        -data: pandas DataFrame containing the data
        -column: String, name of the column to plot the histogram for
        -bins: Integer, number of bins in the histogram
        -title: String, title of the plot
        """
        # Check if the column exists in the DataFrame
        if column not in data.columns:
            print(f"Column '{column}' not found in the DataFrame.")
            return
        
        # Drop NA values and calculate mean and standard deviation
        column_data = data[column].dropna()
        mean, std = np.mean(column_data), np.std(column_data)
    
        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(column_data, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')
        
        # Plotting the normal distribution curve, defininf its range to be between 2 to three SD
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        p = stats.norm.pdf(x, mean, std)
        plt.plot(x, p, 'k', linewidth=2, label= 'Normal Distribution')
        
        plt.title(f"Histogram of {frequency.capitalize()} Returns with Normal Fit")
        plt.xlabel(f'{frequency.capitalize()} Returns')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        
        # Normality Tests
        shapiro_test = stats.shapiro(column_data)
        dagostino_test = stats.normaltest(column_data)
        anderson_test = stats.anderson(column_data)
        print(colored(f"Shapiro-Wilk Test: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}", 'magenta', 'on_grey' ))
        print(colored(f"D'Agostino's K^2 Test: Statistic={dagostino_test[0]}, p-value={dagostino_test[1]}",'magenta', 'on_grey'))
        print("Anderson-Darling Test:")
        for i in range(len(anderson_test.critical_values)):
            sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
            print(colored(f"   {sl}%: {cv}, {'Reject' if anderson_test.statistic > cv else 'Accept'}", 'magenta','on_grey'))
            
def Correlogram(data, lower_bound, upper_bound, titlez='Returns', frequency = 'daily'): 
    data = data.dropna().to_numpy() #Last command makes it so that we have a NumPy array.
    acf(data, alpha= .05,zero=False) # By default is at 5 percent 
    
    plt.ylim(lower_bound, upper_bound)  #Make these user selectable
    plt.title(f"Correlogram of {frequency.capitalize()} {titlez.capitalize()}")
    plt.show()
    

def apply_hill_estimator(data, hill_estimator):
    def function(significance_level):
        left, right = hill_estimator_both_tails(data, significance_level)
        hill_estimator.loc[significance_level, 'Left Tail Index'] = left
        hill_estimator.loc[significance_level, 'Right Tail Index'] = right
    return np.vectorize(function)

def hill_estimator_both_tails(data, tail_percent=0.05):
        data = data.dropna()*100
        n = len(data)
        k = int(n * tail_percent)  # Determine k based on a percentage of the total observations
        # Right Tail (largest values)
        right_tail = np.sort(data)[-k:]  # Ascending sort, -k indicates to select k elements sarting from the end
        right_tail_index = k / (np.sum(np.log(right_tail/right_tail[0])))
        # Left Tail (smallest values)
        left_tail = np.sort(-data[data < 0])[-k:]  # Ascending sort
        left_tail_index = k / (np.sum(np.log(left_tail/left_tail[0])))
        return left_tail_index, right_tail_index


def calculate_hqc(model_result, n):
    log_likelihood = model_result.loglikelihood
    num_params = model_result.num_params
    hqc = -2 * log_likelihood + 2 * num_params * np.log(np.log(n))
    return hqc    
def fit_garch_model(data, p, q):
    model = arch_model(data, p=p, q=q, dist='skewt')
    fit = model.fit(disp='off')
    hqc = calculate_hqc(fit, len(data))
    return p, q, fit.aic, fit.bic, hqc
def garch_lag_selection(data, max_lags=5, max_lags_var=5):
    results = []
    with ThreadPoolExecutor() as executor: #Parallel computing  (Initialization of pool)
        futures = [executor.submit(fit_garch_model, data, p, q) for p in range(1, max_lags + 1) for q in range(1, max_lags_var + 1)]
        for future in as_completed(futures):
            results.append(future.result())

        results_df = pd.DataFrame(results, columns=['p', 'q', 'AIC', 'BIC', 'HQC'])
        results_df['Optimal'] = (results_df['AIC'] == results_df['AIC'].min()) | \
                                (results_df['BIC'] == results_df['BIC'].min()) | \
                                (results_df['HQC'] == results_df['HQC'].min())
    return results_df



def fit_and_forecast_garch(data, fraction, p, q, o, mean, power, vol, dist, hor, frequency = 'daily'):
    """
    Fits a GARCH model on a specified fraction of the provided time series data
    and forecasts for the remaining data.

    Parameters:
    data: Time series data for model fitting and forecasting.
    fraction: Fraction of the data to be used for training (e.g., 0.66 for 2/3).
    p, q, o: Lag order parameters for the GARCH model.
    mean: The mean model specification ('constant', 'zero', etc.).
    power: Power to use for ARCH and GARCH terms.
    vol: Volatility model to use ('GARCH', 'EGARCH', etc.).
    dist: Distribution assumption for the standardized residuals ('normal', 't', 'skewt', etc.).
    hor: the horizon of forecast
    var_levels (list): List of confidence levels for VaR (e.g., [0.01, 0.05]).
    Returns:
    fitted_model: The fitted ARCH model object.
    forecasts: The forecasts from the fitted model.
    var_forecasts (dict): VaR forecasts at specified confidence levels.
    """
    # Splitting the data
    data = (data*100).dropna()
    last_obs = data.index[int(len(data)*fraction)+1]
    model = arch_model(data , p=p, q=q, o=o, mean=mean, power=power, vol=vol, dist=dist)
    fitted_model = model.fit(last_obs=last_obs,update_freq=15)
    
    # Forecasting volatility
    forecasts = fitted_model.forecast(horizon=hor, reindex=False)
    forecasted_values = forecasts.variance
    
    #Forecasting VaR
    std =  (data[:int(len(data)*fraction)] - fitted_model.params["mu"]) / fitted_model.conditional_volatility[:int(len(data)*fraction)]
    
    q = std.quantile([0.01, 0.05])
    m = forecasts.mean.values[:,1]
    n= forecasted_values['h.1'].values
    value_at_risk = -m[:,None] - np.sqrt(n[:,None]) * q.values
    value_at_risk = pd.DataFrame(value_at_risk, columns=["1%", "5%"], index=forecasted_values['h.1'].index)
    ax = value_at_risk.plot(legend=False)
    xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])
    returns = data[int(len(data)*fraction):].copy()
    returns.name = "EURO STOXX 50 Return"
    Exceedence = [] # Will store exceedances so performance can be computed
    
    for idx in value_at_risk.index:
        if returns[idx] > -value_at_risk.loc[idx, "5%"]:
            Exceedence.append("No Exceedence")
        elif returns[idx] < -value_at_risk.loc[idx, "1%"]:
            Exceedence.append("1% Exceedence")
        else:
            Exceedence.append("5% Exceedence")
    Exceedence = np.array(Exceedence, dtype="object")
    ax.set_title(f"{frequency.capitalize()} Simulation VaR")
    leg = ax.legend(frameon=False, ncol=3)

    
    return fitted_model, forecasted_values, value_at_risk, forecasts, Exceedence

def fit_and_forecast_egarch(data, fraction, p,  q, o, mean, dist, hor, frequency='daily'):
    """
    Fits an EGARCH model on a specified fraction of the provided time series data
    and forecasts for the remaining data.

    Parameters:
    data: Time series data for model fitting and forecasting.
    fraction: Fraction of the data to be used for training (e.g., 0.66 for 2/3).
    p, o, q: Lag order parameters for the EGARCH model.
    mean: The mean model specification ('constant', 'zero', etc.).
    dist: Distribution assumption for the standardized residuals ('normal', 't', 'skewt', etc.).
    hor: the horizon of forecast.
    frequency: Frequency of the data ('daily', 'monthly', etc.).

    Returns:
    fitted_model: The fitted ARCH model object.
    forecasts: The forecasts from the fitted model.
    var_forecasts: VaR forecasts at specified confidence levels.
    """
    # Splitting the data
    data = (data * 100).dropna()
    last_obs = data.index[int(len(data) * fraction) + 1]
    model = arch_model(data, p=p, o=o, q=q, mean=mean, vol='EGARCH', dist=dist)
    fitted_model = model.fit(last_obs=last_obs, update_freq=15)
    
    # Forecasting volatility
    forecasts = fitted_model.forecast(start=last_obs, method='simulation', simulations=1000, horizon=hor)
    forecasted_values = forecasts.variance

    # Forecasting VaR
    std = (data[:int(len(data) * fraction)] - fitted_model.params["mu"]) / fitted_model.conditional_volatility[:int(len(data) * fraction)]
    q = std.quantile([0.01, 0.05])
    m = forecasts.mean.values[:, 1]
    n = forecasted_values['h.1'].values
    value_at_risk = -m[:, None] - np.sqrt(n[:, None]) * q.values
    value_at_risk = pd.DataFrame(value_at_risk, columns=["1%", "5%"], index=forecasted_values['h.1'].index)

    # Plotting VaR
    ax = value_at_risk.plot(legend=False)
    ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])
    returns = data[int(len(data) * fraction):].copy()
    returns.name = "EURO STOXX 50 Return"
    exceedence = [] # Will store exceedances for performance evaluation

    for idx in value_at_risk.index:
        if returns[idx] > -value_at_risk.loc[idx, "5%"]:
            exceedence.append("No Exceedence")
        elif returns[idx] < -value_at_risk.loc[idx, "1%"]:
            exceedence.append("1% Exceedence")
        else:
            exceedence.append("5% Exceedence")
    exceedence = np.array(exceedence, dtype="object")
    ax.set_title(f"{frequency.capitalize()} Simulation VaR EGARCH")
    ax.legend(frameon=False, ncol=3)

    return fitted_model, forecasted_values, value_at_risk, forecasts, exceedence
   


def ploty(data,title = '', ylabel =''):

    plt.plot(data)
    plt.title(f'{title}')
    plt.xlabel('Time')
    plt.ylabel(f'{ylabel}')
    plt.show()

def add_double_lines_to_latex(latex_str):
    """
    Post-process a LaTeX table string to add double lines after each \toprule, \midrule, \bottomrule.

    Args:
    - latex_str (str): The original LaTeX table string.

    Returns:
    - str: Modified LaTeX table string with double lines.
    """
    # Replace each booktabs command with its double line version
    latex_str = latex_str.replace('\\toprule', '\\toprule\\toprule')
    latex_str = latex_str.replace('\\midrule', '\\midrule\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\\bottomrule')
    return latex_str

def calculate_empirical_var(returns, confidence_levels=[0.01, 0.05]):
    """
    Calculate the empirical Value at Risk (VaR) at specified confidence levels.

    Parameters:
    returns (pd.Series): Historical returns.
    confidence_levels (list): List of confidence levels to calculate VaR for.

    Returns:
    var_values (dict): VaR values at the specified confidence levels.
    """
    
    sorted_returns = returns.sort_values().dropna()
    var_values = {}

    for level in confidence_levels:
        var_values[level] = sorted_returns.quantile(level)

    return var_values


            