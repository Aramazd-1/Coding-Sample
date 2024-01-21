import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing # Used to do standardization of variables
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from termcolor import colored
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import statsmodels.api as sm
import matlab.engine
from statsmodels.graphics.tsaplots import plot_acf as acf

def adf_test(data, signif = 0.05):
    adf_results = {}
    if isinstance(data, pd.DataFrame):
        columns = data.columns
    elif isinstance(data, np.ndarray):
        columns = range(data.shape[1])
    else:
        raise ValueError("Input data must be a pandas DataFrame or a numpy array.")

    for taker in columns:
        adf_result = adfuller(data[taker].dropna())
        adf_results[taker] = {'ADF Statistic': adf_result[0], 'p-value': adf_result[1]}
        print(f'ADF Statistic for {taker}: {adf_result[0]}')
        print(f'p-value for {taker}: {adf_result[1]}')

        if adf_result[1] > signif:
            print(colored(f"ADF test for {taker} detects a unit root, indicating non-stationarity.", 'red', 'on_grey'))
        else:
            print(colored(f"ADF test for {taker} does not detect a unit root, indicating stationarity", 'green', 'on_grey'))

def Correlogram(data, lower_bound, upper_bound, titlez='Returns', frequency = 'daily'): 
    data = data.dropna().to_numpy() #Last command makes it so that we have a NumPy array.
    acf(data, alpha= .05,zero=False) # By default is at 5 percent 
    
    plt.ylim(lower_bound, upper_bound)  #Make these user selectable
    plt.title(f"Correlogram of {frequency.capitalize()} {titlez.capitalize()}")
    plt.show()
    
def ploty(data, labels, title='',xticks=None, xlabel='', ylabel='', line_styles=None, line_colors=None, grid=False, save_path=None):
    """
    Plots the given data.

    Parameters:
    data (array-like or list of array-like): The data to be plotted.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    line_style (str): Style of the plot line (e.g., '-', '--', '-.', ':').
    line_color (str): Color of the plot line.
    grid (bool): Whether to display a grid.
    save_path (str): Path to save the plot image, if desired.
    """
    plt.figure(figsize=(10, 6))  # Larger figure size
    for i, d in enumerate(data): plt.plot(d, label=labels[i], linestyle=line_styles[i], color=line_colors[i])
    plt.legend()  
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks)
    if grid:
        plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")

    plt.show()
def ploty_old(data,title='', xlabel='', ylabel='', grid=False, save_path=None):

    plt.plot(data)
    plt.title(f'{title}')
    plt.xlabel('Time')
    plt.ylabel(f'{ylabel}')
    if grid:
        plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")

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
# Example usage:
#     file.write(add_double_lines_to_latex(hill_estimator_daily.to_latex(index=True, escape=False, column_format='c c c')))
#     file.write('\n\n')

def simulate_series(interval, n_observations, series_names, kwargs_list, distribution_function=np.random.normal, mean=0, standardize = 1, std_devs=None,start="2023-01-01"):    
    """
    Generates time series data for multiple series based on specified distribution parameters and characteristics.

    Parameters:
        interval (str): The time interval between observations (e.g., 'D' for daily, 'Q' for quarterly).
        n_observations (int): The number of observations to generate.
        series_names (list): A list of names for the generated series.
        kwargs_list (list): A list of dictionaries, where each dictionary contains distribution-specific parameters
                            for each series. These parameters include the distribution's parameters (e.g., degrees of
                            freedom for Student's t-distribution) and any additional parameters for adjusting mean and
                            standard deviation.
        distribution_function (function, optional): The probability distribution function used to generate data for each
                            series. Default is np.random.normal, representing a normal distribution.
        mean (float, optional): The mean value to be applied to the generated data for each series. Default is 0.
        start (str, optional): The start date for the time series data. Default is "2023-01-01."
        Standardize (bool) : If set to true standardization is performed o nthe dataset
    Returns:
        pd.DataFrame: A pandas DataFrame containing time series data. Each column in the DataFrame represents a series
                      with the specified name.
    """
    index = pd.date_range(start, periods=n_observations, freq=interval)
    data = {name: distribution_function(**kwargs, size=n_observations) for name, kwargs in zip(series_names, kwargs_list)}
    #zip is a built-in Python function that takes two or more sequences and aggregates them into a single iterator of tuples. Each tuple contains elements from each of the input sequences, matched based on their position.
    data = pd.DataFrame(data, index=index)
    if standardize == True:
        data = preprocessing.scale(data)
    
    time_series_df = pd.DataFrame(data, index=index, columns= series_names)

    return time_series_df
#Example for a Normal (for a student t just remove both parameters and place df instead)
# kwargs_list = [
#     {'loc': 0, 'scale': 1},  # Parameters for Series1
#     {'loc': 5, 'scale': 2}   # Parameters for Series2
# ]

def analyze_pca(data,threshold = 0.95, n_components = 1,standardize=0):
    """
    Performs PCA analysis on the given DataFrame and returns the optimal number of components.

    Parameters:
    - data (pd.DataFrame): The DataFrame on which PCA will be performed.
    - ncomponents (int): Specifies the number of principal component for PCA
    - threshold (float): Thresshold of explained variance needed in order to find the optial number of components
    Returns:
    - pca: Object 
    - Principal_components: array of float64 containing principal components
    """
    if standardize == True:
        std_data=StandardScaler().fit_transform(data)
        data = pd.DataFrame(std_data, index=data.index, columns=data.columns)
        
    pca = PCA(n_components)
    Principal_components = pca.fit_transform(data)
    pca_fit= pca.fit(data)
    explained_variance = pca_fit.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)


    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual explained variance')
    plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.text(0.5, threshold, f'{threshold*100}% threshold', color = 'red', va='bottom', ha='right')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.xticks(PC_values)
    plt.show()
    return pca, Principal_components

def Companion_Matrix_compute(AR_Matrices, M):
    """
    Parameters
    ----------
    AR_Matrices : The matrices with the autoregressive coefficients for each lag. 
    They are assumed to be organized as a dictionary containing 2D arrays (the matrices)
    entry : the entry where the first matrix is located. Will be used to ocmpute the number of variables present in the system

    Returns
    -------
    Companion_Matrix : A 2D array Companion matrix

    """
    
    p= len(AR_Matrices) #number of lags
    if p>1:
        top_row = np.hstack([AR_Matrices[f"AR{i+1}"] for i in range(p)])
        bottom_part = np.hstack([np.eye(M * (p - 1)), np.zeros((M * (p - 1), M))])
        Companion_Matrix = np.vstack([top_row, bottom_part])
    else: 
        top_row = np.hstack([AR_Matrices[f"AR{i+1}"] for i in range(p)])
        Companion_Matrix = top_row
    return Companion_Matrix
    
def histogram_Normality_test(data, series, column='Return', bins=50, frequency='daily'):
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
        
        plt.title(f"Histogram of {frequency.capitalize()} {series.capitalize()} with Normal Fit")
        plt.xlabel(f'{frequency.capitalize()} {series.capitalize()}')
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

def breusch_test(Residuals, p ):
    """
    Performs a Beusch_Godfrey test on the given Residuals considering the given number of lags
    """
    for _ in Residuals.columns: 
        lm_test_statistic, p_value, f_test_statistic, f_p_value = acorr_breusch_godfrey(Residuals[_], nlags=p)
        print(f"Results for {_}:")
        print(f"Lagrange Multiplier statistic: {lm_test_statistic}")
        print(f"p-value: {p_value}")
        print(f"f-statistic: {f_test_statistic}")
        print(f"f-test p-value: {f_p_value}\n")

# Functions to implement a moving block bootstrap
def generate_blocks(residuals, block_length):
    n = len(residuals)
    return [residuals[i:i+block_length] for i in range(n-block_length+1)]
def select_blocks(blocks, n_blocks):
    """
    Randomly selects blocks from a list containing numpy arrays and gives back data ina time series format
    It's useful in MBB

    Parameters
    ----------
    blocks : A list containing numpy arrays
    n_blocks : The number of blocks

    Returns
    -------
    stacked_array : The arrays stacked together in a time series format.

    """
    selected_indices = np.random.choice(len(blocks), size=n_blocks, replace=False)
    selected_arrays = [blocks[i] for i in selected_indices]
    stacked_array = np.vstack(selected_arrays)
    return stacked_array
    
def compute_irf(IRF_matrices_boot, hor, Var_data_columns, percentiles=[5, 16, 84, 95]):
    """
    Computes statistics for bootstrapped IRF matrices.

    Parameters:
    - IRF_matrices_boot (numpy.ndarray): 3D array of bootstrapped IRF matrices.
    - hor (int): The horizon for the IRF calculation.
    - Var_data_columns (List[str]): List of column names for the DataFrame.
    - percentiles (List[int]): List of percentiles to compute (default is [5, 16, 84, 95]).

    Returns:
    - dict: A dictionary of pandas DataFrames containing the calculated statistics.
    """

    statistics = {}
    for percentile in percentiles: statistics[f'percentile_{percentile}'] = []
    statistics['average'] = []

    for h in range(hor + 1):
        statistics['average'].append(np.mean(IRF_matrices_boot[:, h, :], axis=0))
        for percentile in percentiles:
            statistics[f'percentile_{percentile}'].append(np.percentile(IRF_matrices_boot[:, h, :], percentile, axis=0))

    # Convert lists of results to pandas DataFrames
    for key in statistics: statistics[key] = pd.DataFrame(np.vstack(statistics[key]), index=range(0, hor+1), columns=Var_data_columns)

    return statistics   

