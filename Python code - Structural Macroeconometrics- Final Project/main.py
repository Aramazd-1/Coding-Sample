#### Structural Macroeconometrics Final Project ####
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
import time
import numpy as np
import statsmodels.api as sm
from numpy.linalg import matrix_power, inv
from functools import partial

# User defined functions
import Structural_Functions
from VAR_estimation import estimate_var_model_const_trend, estimate_var_model_const_trend_ols_boot, VAR_lag_selection
from Bootstrap import MBB_parallel, exec_MBB_parallel

cell_time = {}   # Initialize dictionary for storing time taken to run each cell


@contextmanager
def timer(cell_name):
    start_time = time.time()
    yield  # Needed to pause execution of timer and then restart it after code is done
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    cell_time[cell_name] = round(elapsed_time, 2)


# %% # Loading and processing climate data:
# 1. Load climate data for Italy and Croatia from Excel files.
# 2. Drop irrelevant columns ('fire', 'hail', 'E3CI') for Italy data and compute mean of remaining columns.
# 3. Insert a new 'E3CI' column in Italy data based on the computed mean.
# 4. Perform similar operations for Croatia climate data to use as an instrument.
# 5. Conduct a preliminary analysis of the data including mean and standard deviation.
# 6. Apply PCA (Principal Component Analysis) to both Italy and Croatia data.
# 7. Extract and analyze eigenvalues and loadings from PCA.
# 8. Insert PCA-based E3CI values as new columns in the data.
# 9. Organize the processed data into a dictionary for easy access and housekeeping.

path = "D:\Study\Github\Coding-Sample\Python code - Structural Macroeconometrics- Final Project\Data"

climate_data = pd.read_excel(f'{path}\E3CI_data.xlsx', 'Italy_comp_E3CI', index_col=0)
climate_data = climate_data.drop('fire', axis=1).drop('hail',
                                                      axis=1)  # We drop those 2 to conform withne IFAB approach an drop E3CI since it is computed on an older value
climate_data = climate_data.drop('E3CI', axis=1)  # We drop the original E3CI since it is computed on an older value
E3CI_mean = climate_data.mean(axis=1)
climate_data.insert(0, "E3CI", E3CI_mean)
# insert one of these in the code below to load the instrument you wish to use
climate_data_instrument = pd.read_excel(f'{path}\E3CI_data.xlsx', 'Croatia_comp_E3CI', index_col=0).drop('fire',
                                                                                                         axis=1).drop(
    'hail', axis=1)
print(
    f"Mean\n{climate_data.mean(axis=0)}\nStandardDeviation:\n{climate_data.std(axis=0)}")  # Manual inspection of data to check if they are standardized or not

# PCA
PC, Principal_Components = Structural_Functions.analyze_pca(climate_data.drop('E3CI', axis=1), threshold=0.90,
                                                           n_components=4, standardize=0)
eigenvalues_Italy = PC.explained_variance_
loadings = PC.components_
print(f'these are{loadings}\n these are eigenvalues {eigenvalues_Italy}')

E3CI_PCA = Principal_Components[:, 0]
# E3CI_PCA = E3CI_mean #quick command so that if you uncomment you can check what happens when using the mean
E3CI_PCA = pd.DataFrame(E3CI_PCA, index=climate_data.index, columns=['E3CI_PCA'])
climate_data.insert(0, "E3CI_PCA", E3CI_PCA)
print(f"Mean\n{climate_data['E3CI_PCA'].mean(axis=0)}\nStandardDeviation:\n{climate_data['E3CI_PCA'].std(axis=0)}")

## We now use PCA to construct our instrument ##
PC_instrument, Principal_Components_instrument = Structural_Functions.analyze_pca(climate_data_instrument,
                                                                                  threshold=0.90, n_components=4,
                                                                                  standardize=0)
eigenvalues_Italy = PC_instrument.explained_variance_
loadings_instrument = PC_instrument.components_
print(f'these are{loadings_instrument}\n these are eigenvalues {eigenvalues_Italy}')
E3CI_PCA_instrument = Principal_Components_instrument[:, 0]
# E3CI_PCA_instrument = climate_data_instrument.mean(axis=1)
# #quick command so that if you uncomment you can check what happens when using the mean
E3CI_PCA_instrument = pd.DataFrame(E3CI_PCA_instrument, index=climate_data_instrument.index,
                                   columns=['E3CI_PCA_instrument'])
climate_data.insert(0, "E3CI_PCA_instrument", E3CI_PCA_instrument)

# Housekeeping and cleaning
climate_dataframes = {}
climate_dataframes.update({'climate_data': climate_data, 'climate_data_instrument': climate_data_instrument,
                           'loadings': [loadings, loadings_instrument]})
del climate_data, climate_data_instrument, E3CI_mean, PC, PC_instrument, Principal_Components, Principal_Components_instrument, loadings, loadings_instrument
# %% Import and clean data macro ###
# note we are trying to make sure that all our data is in dataframes here!
#### User can basically skip this part and directly load Var_data ####
# Data loading, processing, and preparation for VAR analysis:
# 1. Set start and end dates for the analysis period.
# 2. Load and preprocess macroeconomic data from EUROSTAT, including seasonal decomposition and differencing.
# 3. Load and preprocess additional macroeconomic data (excluding energy uncertainty data).
# 4. Load and preprocess ECB data, including seasonal and trend decomposition, and logarithmic differencing.
# 5. Load and preprocess heat and gas price data, including seasonal adjustment and logarithmic differencing.
# 6. Concatenate various data sources into a single DataFrame for VAR analysis, and insert PCA-based E3CI and net gas prices.
# 7. Export the prepared VAR data and PCA-instrument data to Excel files.
# 8. Load and preprocess weak instrument data (global sea level) for optional use in SVAR analysis.
# 9. Decide between using weak or strong instrument based on 'use_weak' flag and adjust the data accordingly.
# 10. Perform housekeeping by deleting unused variables to free up memory.

start_date = pd.to_datetime('1997-07-01')
end_date = pd.to_datetime('2023-06-30')
dates = {}
dates.update({'start_date': start_date, 'end_date': end_date})

df_macro_eurostat = pd.read_excel(f"{path}\Working Data_EUROSTAT.xlsx", index_col=0).dropna().drop('Euro_Short_Rate',
                                                                                                   axis=1).astype(float)
df_macro_eurostat.index = pd.to_datetime(df_macro_eurostat.index, format='%Y-%m')
decomposition = sm.tsa.seasonal_decompose(df_macro_eurostat['HICP_FE'], model='multiplicative', period=12).seasonal
df_macro_eurostat['HICP_FE'] = df_macro_eurostat['HICP_FE'].div(decomposition, axis=0)
df_macro_eurostat = np.log(df_macro_eurostat).diff(
    periods=1).dropna() * 100  # year over year growth of non food and energy inflation
df_macro_eurostat = df_macro_eurostat[(df_macro_eurostat.index >= start_date) & (df_macro_eurostat.index <= end_date)]

# ♥loads energy_uncertainty_italy, macro unc and iprod
df_macro_else = pd.read_excel(f"{path}\Working Data_Else.xlsx", index_col=0).drop('energy_uncertainty', axis=1)
df_macro_else.index = pd.to_datetime(df_macro_else.index, format='%Y-%m')
df_macro_else = df_macro_else[
    (df_macro_else.index >= start_date) & (df_macro_else.index <= end_date)]  # .drop('energy_uncertainty',axis=1)

df_ECB = pd.read_excel(f"{path}\Working Data_ECB.xlsx", 'B', index_col=0)
df_ECB.index = pd.to_datetime(df_ECB.index, format='%Y-%m')
decomposition = sm.tsa.seasonal_decompose(df_ECB['HICP'], model='multiplicative', period=12)
df_ECB['HICP'] = (df_ECB['HICP'].div(decomposition.seasonal, axis=0)).div(decomposition.trend,
                                                                          axis=0)  # we also detrend this inflation
df_ECB.loc[:, df_ECB.columns != '3monthrate'] = df_ECB.loc[:, df_ECB.columns != '3monthrate'].apply(
    lambda x: (np.log(x).diff(periods=12)) * 100)  # Difference over precedent year of GDP index and HICP
df_ECB['3monthrate'] = df_ECB['3monthrate'].diff(periods=1)  # first difference of 3 month rate
df_ECB = df_ECB[(df_ECB.index >= start_date) & (df_ECB.index <= end_date)]
df_ECB.index = df_macro_else.index

heatgas_prices = (pd.read_excel(f'{path}\heatgas_prices.xlsx', index_col=0))
decomposition = sm.tsa.seasonal_decompose(heatgas_prices, model='additive', period=12)
heatgas_prices = heatgas_prices - decomposition.seasonal.values.reshape(-1, 1)
heatgas_prices = np.log(pd.read_excel(f'{path}\heatgas_prices.xlsx', index_col=0)).diff(periods=1).dropna()
heatgas_prices.index = pd.to_datetime(heatgas_prices.index, format='%Y-%m')
heatgas_prices = heatgas_prices[(heatgas_prices.index >= start_date) & (heatgas_prices.index <= end_date)]

Var_data = pd.concat([df_macro_eurostat, df_macro_else, df_ECB], axis=1).dropna()
E3CI_PCA = E3CI_PCA[(E3CI_PCA.index >= start_date) & (E3CI_PCA.index <= end_date)]
E3CI_PCA_instrument = E3CI_PCA_instrument[
    (E3CI_PCA_instrument.index >= start_date) & (E3CI_PCA_instrument.index <= end_date)]
Var_data.insert(0, "E3CI_PCA", E3CI_PCA), Var_data.insert(1, 'net_gasprices', heatgas_prices)
Var_data = Var_data.drop('Uncertainty_Italy_News', axis=1).drop('Iprod_Italy_Year_on_year_growth',
                                                                axis=1)  # .drop('HICP_FE', axis= 1)
Var_data.to_excel(
    excel_writer='D:\Study\Github\Coding-Sample\Python code - Structural Macroeconometrics- Final Project\Data\Var_data.xlsx',
    index=False)
E3CI_PCA_instrument.to_excel(
    excel_writer='D:\Study\Github\Coding-Sample\Python code - Structural Macroeconometrics- Final Project\Data\instrument.xlsx',
    index=False)

# Now we load the instrument
use_weak = 0  #### IF set to 1 the weak instrument will be used in the SVAR, if 0 the strong one will

sea_level_data = pd.read_excel(f'{path}\Global _Sea Level.xlsx', index_col=0)  # Load the weak instrument
sea_level_data.index = pd.to_datetime(sea_level_data.index)
instrument_weak = sea_level_data.resample('M').mean()  # Resample the data to monthly frequency and calculate the mean
instrument_weak = instrument_weak.diff(periods=1).dropna()
instrument_weak = instrument_weak[(instrument_weak.index >= start_date) & (instrument_weak.index <= end_date)]

if use_weak == 0:
    instrument = E3CI_PCA_instrument
    print(f'Start date: {start_date}')
    print(f'End date : {end_date}')
else:
    instrument = instrument_weak
    Var_data = Var_data[(Var_data.index >= start_date) & (
            Var_data.index <= instrument.index.max())]  # We trim Var_data so it matches the instrument
    print(f'Start date: {start_date}')
    print(f'End date : {instrument.index.max()}')

# Housekeeping and cleaning
del df_macro_eurostat, df_macro_else, start_date, end_date, heatgas_prices, decomposition, df_ECB
# %% Preliminary Plotting
#In this cell, preliminary plotting of all the variables is performed
for _ in climate_dataframes['climate_data'].columns: Structural_Functions.ploty_old(
    climate_dataframes['climate_data'][f"{_}"],
    title=f"{_}")  # Plotting loop
for _ in Var_data.columns: Structural_Functions.ploty_old(Var_data[f"{_}"], title=f"{_}")  # Plotting loop
# %% This cell uses matlab with MLIK, slower but gives same results and in a dataframe. Hence it's recommended for readability but takes a bit. One can also directly use the other function in the cell below, made array like.
# Actually, they give a a small difference in the value of the constant, but it doesn't matter at all since everything else is equal, including the residuals
# import matlab.engine
# with timer('Var cell'):
#     adf_test(Var_data) #We run an ADF test on the variables in our dataset to check for their stationarity
#     eng = matlab.engine.start_matlab() #The bulk of the time here is due to matlab boot time
#     criteria_df = VAR_lag_selection(Var_data,eng, max_p=6)
#     Non_ar_params, Sigma_u, AR_matrices, Residuals = estimate_var_model_const_trend(Var_data,eng, p=2) #We use MATLAB to estimate a VAR
#     eng.quit()
#     print(add_double_lines_to_latex(criteria_df.to_latex(index=True, escape=False, column_format='c c c c'))) #To manually export the table to latex using console, file.write instead of print if you prefer
# %%
#This cell performs:
# 1. ADF test on all the variables present in VAR_data
# 2. Estimate a VAR model on VAR_data with p=2, set it to other values if you wish to test
#Notice that lag selection has been currently implemented for the matlab model (cell above). 
#One could also employ the automatic best lag selection from python
with timer('Var cell'):
    Structural_Functions.adf_test(Var_data)
    Sigma_u, AR_matrices, Residuals, Non_ar_params = estimate_var_model_const_trend_ols_boot(Var_data, p=2,
                                                                                             noprint=False,
                                                                                             returnconst=True)  # If norpint is false then summary and correlation matrix of residuals is printed
# %% Structural VAR analysis and Impulse Response Function (IRF) computation:
    # 1. Determine the number of variables, effective sample size, number of lags, horizon for IRFs, and number of estimated parameters.
    # 2. Optionally switch instruments by setting 'use_weak' and uncommenting relevant lines.
    # 3. Validate the length of the VAR dataset against the instrument.
    # 4. Compute and print standard deviation of the instrument.
    # 5. Perform Augmented Dickey-Fuller (ADF) test on the instrument.
    # 6. Compute Companion Matrix from Autoregressive (AR) matrices.
    # 7. Plot eigenvalues of the Companion Matrix with respect to the unit disk.
    # 8. Calculate the sample covariance between the instrument and the residuals, and the corresponding statistics.
    # 9. Identify the direction of the shock in the VAR model.
    # 10. Select the response matrix based on the number of lags.
    # 11. Compute the Impulse Response Functions (IRFs) for the defined horizon and store in a DataFrame.
    # 12. Compile structural parameters for further analysis or reporting.
    # 13. Perform housekeeping by deleting temporary variables to conserve memory. 
# M = AR_matrices["AR1"].shape[0] # Number of variables if you used matlab
M = AR_matrices[0].shape[0]  # Number of variables
T = len(Var_data) - len(AR_matrices)  # Effective sample
p = len(AR_matrices)  # number of lags
hor = 24  # The horizon of our IRFs
k = M * 2 + M * M * p  # Number of estimated parameters (m*2 constants, trend+intercept) and therest are the AR matrices M*M for each p

# If you want to try out warm as an instrument choose use_weak = 0 and uncomment the two lines below:
# instrument = climate_dataframes['climate_data_instrument'].iloc[198:,4]
# instrument= pd.DataFrame(instrument, index = Var_data.index)

if len(Var_data) == len(instrument):
    print('Var dataset has same length as instrument')
else:
    raise ValueError(
        f"Var dataset doesn't have the same length as the instrument, please revise dates by {len(instrument) - len(Var_data)}")

print(f'std dev:{np.std(instrument)} ')
Structural_Functions.adf_test(instrument)


Companion_Matrix = Structural_Functions.Companion_Matrix_compute(AR_matrices, M)
eigenvalues = Structural_Functions.plot_eigenvalues_with_unit_disk(Companion_Matrix)

psi_hat = 1 / (T - p) * np.dot(Residuals.T, instrument.iloc[
                                            p:])  # We compute the sample covariance between the instrument and the Residuals
rhosqu_hat = psi_hat.T @ np.linalg.inv(Sigma_u) @ psi_hat

rho_hat_std = (rhosqu_hat ** (1 / 2))
corr_instr = rho_hat_std / (np.std(instrument.iloc[p:]).item())
print(corr_instr)
k_hat = rhosqu_hat ** (-1 / 2) * psi_hat

if k_hat[0] < 0: k_hat = k_hat * (-1)  # We identify a one standard deviation increase in the target variable
if p > 1:
    R_selection = np.hstack([np.eye(M), np.zeros((M, M * (p - 1)))])
else:
    R_selection = np.eye(M)

IRF_hat_matrix = []
for h in range(0, hor + 1):
    IRF_hat_h = (R_selection @ matrix_power(Companion_Matrix, h) @ R_selection.T @ k_hat).T
    IRF_hat_matrix.append(IRF_hat_h)

IRF_hat_matrix = pd.DataFrame(np.vstack(IRF_hat_matrix), index=range(0, hor + 1), columns=Var_data.columns.tolist())
Structparams = [k_hat, corr_instr, rho_hat_std, rhosqu_hat, psi_hat, eigenvalues, Companion_Matrix]

del k_hat, corr_instr, rho_hat_std, rhosqu_hat, psi_hat, eigenvalues, Companion_Matrix, IRF_hat_h
# %% Bootstrap cell (safe and easy, with vectorization)
##############################################################################################################################################################################
############ README: choose whether to run this cell/bootstrap procedure or the one below. This one is widely compatible but slower. They yield the same results  ############
##############################################################################################################################################################################
# Bootstrap procedure for Impulse Response Functions (IRFs) in VAR analysis:
    # 1. Initialize the bootstrap sample with the first 'p' values from the original VAR data.
    # 2. Define a Moving Block Bootstrap (MBB) function to perform the bootstrap process.
    # 3. Inside the MBB function:
    #    - Select blocks of residuals randomly and center them.
    #    - Generate the bootstrap sample by iterating over time points and applying AR model equations.
    #    - Estimate the VAR model parameters and compute sample covariance between instrument and residuals for the bootstrap sample.
    #    - Calculate the corresponding statistics and adjust the direction of shock if necessary.
    #    - Compute the Companion Matrix and IRFs for each bootstrap iteration.
    # 4. Set up external parameters for the bootstrap (block length, number of iterations).
    # 5. Prepare the residuals and initiate the bootstrap process, storing results in a 3D array.
    # 6. Optionally, use parallel processing for the bootstrap procedure.
    # 7. Perform housekeeping by deleting temporary variables to optimize memory usage.

Sample_boot = np.zeros((T + p, M))  # We Initialize the sample
Sample_boot[:p, :] = Var_data.iloc[:p].to_numpy()  # makes first p values equal to the original sample

#The function is left here so closer inspection is possible
def MBB(b,IRF_matrices_boot):  # this is the function that will perform the bootstrap. If you prefer you can run a parallel version of it by using the section below.
    if yesprint == True:
        if (b + 1) % 50 == 0: print(f'botstrap iteration: {b + 1}')
    Residuals_boot = Structural_Functions.select_blocks(blocks, int(T / block_length) + 1)[:T,
                     :]  # Randomly samples block with replacement
    Residuals_boot = Residuals_boot - np.mean(Residuals_boot, axis=0, keepdims=True)  # We center residuals
    instrument_boot = Residuals_boot[:, M]  # Selects the last column to be the Residuals of the instrument
    Residuals_boot = Residuals_boot[:, :M]

    for i in range(p, T): Sample_boot[i] = Non_ar_params[:, 0].T + sum(
        AR_matrices[j] @ Sample_boot[i - j - 1].T for j in range(0, p)) + Residuals_boot[i - 2, :]

    # Estimation of the model and IRF
    Sigma_u_boot, AR_matrices_boot, Residuals_boot = estimate_var_model_const_trend_ols_boot(Sample_boot, p=p,
                                                                                             noprint=True)
    psi_hat_boot = 1 / (T - p) * np.dot(Residuals_boot.T,
                                        instrument_boot)  # We compute the sample covariance between the instrument and the Residual
    rho_squared_hat_boot = psi_hat_boot.T @ np.linalg.inv(Sigma_u_boot) @ psi_hat_boot
    k_hat_boot = rho_squared_hat_boot ** (-1 / 2) * psi_hat_boot
    if k_hat_boot[0] < 0: k_hat_boot = k_hat_boot * (-1)
    Companion_Matrix_boot = Structural_Functions.Companion_Matrix_compute(AR_matrices_boot, M)

    IRF_hat_matrix_boot = np.zeros((hor + 1, M))
    for h in range(0, hor + 1): IRF_hat_matrix_boot[h, :] = (
            R_selection @ matrix_power(Companion_Matrix_boot, h) @ R_selection.T @ k_hat_boot).T
    IRF_matrices_boot[b, :, :] = np.vstack(IRF_hat_matrix_boot)


with timer('Boot cell2'):
    if isinstance(Non_ar_params, pd.DataFrame):
        Non_ar_params = Non_ar_params.to_numpy()
    yesprint = True
    # Initializing external parameters
    block_length = 12
    Residuals_expanded = np.hstack((Residuals, instrument[p:]))  # we stack all residuals together to jointly boot them
    blocks = Structural_Functions.generate_blocks(Residuals_expanded, block_length=block_length)
    nboot = 20000  # Number of boot iterations
    IRF_matrices_boot = np.zeros((nboot, hor + 1, M))  # 3D array to store Bootstrap results
    nboots = np.arange(0, nboot)
    parFunc = partial(MBB, IRF_matrices_boot=IRF_matrices_boot)
    apply = np.vectorize(parFunc)
    apply(nboots)
    del nboot, nboots, blocks, block_length, Residuals_expanded
    # %% Parallel bootstrap cell
    # Parallel bootstrap procedure for Impulse Response Functions (IRFs) in VAR analysis (Optional):
    # WARNING: Manual execution recommended due to potential interpreter issues with "run this cell" commands.
    # 1. The code is set up for parallel execution of the bootstrap process to enhance computational efficiency.
    # 2. Initialize parameters like block length and number of bootstrap iterations.
    # 3. Stack residuals and the instrument for joint bootstrapping.
    # 4. Initialize the 3D array for storing bootstrap results and the bootstrap sample.
    # 5. Use the 'exec_MBB_parallel' function to execute the Moving Block Bootstrap (MBB) in parallel.
    # 6. After completion, stack the results into the IRF_matrices_boot array.
    # 7. Uncomment and run this code section manually for faster execution, especially when using a large number of bootstrap iterations.
    #    Use methods like Ctrl+F9 in Spyder or copy-paste into the console.
    # 8. This method is recommended for users who need quicker computation, while others can use the previously defined numpy vectorization.
    
    # with timer('Boot cellparallel'):
    #     block_length = 12
    #     Residuals_expanded = np.hstack((Residuals, instrument[p:]))  # we stack all residuals together to jointly boot them
    #     blocks = generate_blocks(Residuals_expanded, block_length=block_length)
    #     nboot = 20000 # Number of boot iterations, 1000 is set for a quick check, replace with 20000 to replicate results
    #     IRF_matrices_boot = np.zeros((nboot, hor + 1, M))  # 3D array to store Bootstrap results
    #     Sample_boot = np.zeros((T + p, M))  # We Initialize the sample
    #     Sample_boot[:p, :] = Var_data.iloc[:p].to_numpy()  # makes first p values equal to the original sample
    #     if __name__ == '__main__':
    #         results = exec_MBB_parallel(T, p, M, blocks, block_length, Sample_boot, Non_ar_params, AR_matrices, hor, nboot)
    #     print('hello there: bootstrap completed')
    #     IRF_matrices_boot = np.stack (results, axis=0)

    # %% IRF matrices construction
    # Analysis and visualization of bootstrapped Impulse Response Functions (IRFs):
        # 1. Initialize arrays to store the bootstrap mean, various confidence intervals (95%, 90%, 68%, 5%, 10%).
        # 2. Calculate these statistics for each horizon (h) by averaging and finding percentiles from the bootstrap IRF matrices.
        # 3. Convert the calculated statistics into DataFrames for easy handling.
        # 4. Plot IRFs for each variable in the VAR model:
        #    - Include original estimate, mean, and different percentile-based confidence intervals in the plot.
        #    - Customize the plot with labels, line styles, and colors for clarity and distinction.
        # 5. Use the custom 'ploty' function from the 'Structural_Functions' module for plotting.
        # 6. Perform housekeeping by deleting temporary variables and clearing memory.


    avg_IRF_hat_matrix_boot = np.zeros((hor + 1, M))  # Bootstrap mean to check for bias
    top_IRF_hat_matrix_boot = np.zeros((hor + 1, M))  # 95 percent confidence interval
    top68_IRF_hat_matrix_boot = np.zeros((hor + 1, M))
    bot68_IRF_hat_matrix_boot = np.zeros((hor + 1, M))
    top90_IRF_hat_matrix_boot = np.zeros((hor + 1, M))
    bot10_IRF_hat_matrix_boot = np.zeros((hor + 1, M))
    bot_IRF_hat_matrix_boot = np.zeros((hor + 1, M))  # 5 percent confidence interval
    for h in range(0, hor + 1):
        avg_IRF_hat_matrix_boot[h, :] = np.mean(IRF_matrices_boot[:, h, :],
                                                axis=0)  # Averaging the h row across all nboot matrices
        top_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 95, axis=0)
        bot_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 5, axis=0)
        top68_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 84, axis=0)
        bot68_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 16, axis=0)
        top90_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 90, axis=0)
        bot10_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 10, axis=0)

    avg_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(avg_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                           columns=Var_data.columns.tolist())
    top_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                           columns=Var_data.columns.tolist())
    bot_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                           columns=Var_data.columns.tolist())
    top68_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top68_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                             columns=Var_data.columns.tolist())
    bot68_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot68_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                             columns=Var_data.columns.tolist())
    top90_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top90_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                             columns=Var_data.columns.tolist())
    bot10_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot10_IRF_hat_matrix_boot), index=range(0, hor + 1),
                                             columns=Var_data.columns.tolist())

    for _ in Var_data.columns:
        data_to_plot = [
            IRF_hat_matrix[_],
            avg_IRF_hat_matrix_boot[_],
            top_IRF_hat_matrix_boot[_],
            bot_IRF_hat_matrix_boot[_],
            top68_IRF_hat_matrix_boot[_],
            bot68_IRF_hat_matrix_boot[_],
            top90_IRF_hat_matrix_boot[_],
            bot10_IRF_hat_matrix_boot[_]
        ]
        labels = ['Estimate', 'Mean', '95th Percentile', '5th Percentile', '84th percentile', '16th percentile',
                  '90th percentile', '10th percentile']
        line_styles = ['-', '-', '--', '--', '--', '--', '--', '--']
        line_colors = ['blue', 'green', 'red', 'red', 'orange', 'orange', 'purple', 'purple']

        Structural_Functions.ploty(data_to_plot, labels, title=f"IRF for {_}", xticks=range(hor + 1), xlabel="Horizon",
                                   ylabel="Response",
                                   line_styles=line_styles, line_colors=line_colors, grid=True)
del h, line_styles, labels, line_colors, data_to_plot, avg_IRF_hat_matrix_boot, top_IRF_hat_matrix_boot, bot_IRF_hat_matrix_boot, top68_IRF_hat_matrix_boot, bot68_IRF_hat_matrix_boot, top90_IRF_hat_matrix_boot, bot10_IRF_hat_matrix_boot
# %% Local projections
############################################################################################################
# Projection and visualization of IRFs with confidence intervals:
    # 1. Set the horizon for IRFs and initialize grids for different instruments (weak/strong).
    # 2. Convert key dataframes (E3CI_PCA, instrument, Var_data) to numpy arrays for efficient processing.
    # 3. Implement grid testing to determine p-values at different horizons for each variable in the model.
    # 4. Construct regression models for each grid value, horizon, and variable, focusing on the instrument's impact.
    # 5. Calculate and store p-values from these regressions.
    # 6. Initialize arrays to store bounds for 90% and 68% confidence intervals, and the largest p-values.
    # 7. Calculate the confidence intervals at each horizon for each variable using the grid testing results.
    # 8. Plot IRFs with 90% and 68% confidence intervals for each variable:
    #    - Use custom 'plot_irf_with_confidence_intervals' function for visualizing IRFs with confidence bounds.
    #    - Additionally, plot the central points (maximum p-value) and different percentile bounds using 'ploty' function.
    # 9. Customize the plots with labels, colors, and line styles for clarity.
    # 10. Perform housekeeping by deleting temporary variables and clearing memory.

with timer('projection cell'):
    hor = 24  # The horizon of our IRFs
    horizons = np.arange(1, hor + 1)
    if use_weak ==0: grid = np.arange(-1, 1.2, 0.001)
    elif  use_weak ==1 : grid = np.arange (-5,5,0.10)
    # we compute a first regression with instrument
    E3 = E3CI_PCA.to_numpy()  # we use this to reconstruct the actual sample of the regression
    Z = instrument.to_numpy()  # we make the instrument a 2d array
    VD = Var_data.to_numpy()  # we turn var data into numpy array for faster execution
    print(len(grid))
    # Grid testing (A future plan is to make it a function, I'm short on time though)
    p_val = np.zeros((len(grid), hor + 1, M))  # Initialize matrix to store loop results
    for i in np.arange(0, len(grid)):
        for y in np.arange(0, M):
            for h in np.arange(0, hor + 1):
                Y = VD[h + p:, y] - VD[p:len(VD) - h, 0] * grid[i]
                instrument_LP = Z[p:len(VD) - h].reshape(-1, 1)
                instrument_LP_lag1 = Z[p - 1:len(VD) - h - 1].reshape(-1, 1)
                instrument_LP_lag2 = Z[p - 1:len(VD) - h - 1].reshape(-1, 1)

                Y_lag1 = VD[p - 1:T + 1 - h, y].reshape(-1, 1)
                Y_lag2 = VD[p - 2:T - h, y].reshape(-1, 1)

                X_vars_lag1 = np.delete(VD[p - 1:T + 1 - h, :], [0, y], axis=1)
                X_vars_lag2 = np.delete(VD[p - 2:T - h, :], [0, y], axis=1)

                const = np.ones((len(Y), 1))
                trend = np.arange(p + 1, len(Y) + p + 1).reshape(-1,
                                                                 1)  # creates the constant as a list of integers increasing by 1 at a time.

                X = np.hstack([X_vars_lag1, X_vars_lag2, Y_lag1, Y_lag2,
                               trend, const,
                               instrument_LP])  # this makes it so that instrument is at the column M-1 or M-2, implying also his coefficient will be.
                model = sm.OLS(Y, X).fit(cov_type='HC3')

                contrast_matrix = np.zeros(len(model.params))
                contrast_matrix[len(model.params) - 1] = 1
                t_test = model.t_test(contrast_matrix)
                p_value = t_test.pvalue.item()
                p_val[i, h, y] = p_value

    # we now move on to plot the implied IRFs

    lower_bounds_90 = np.zeros((hor + 1, M))  # ♣We initialize the arrays
    upper_bounds_90 = np.zeros((hor + 1, M))
    lower_bounds_68 = np.zeros((hor + 1, M))
    upper_bounds_68 = np.zeros((hor + 1, M))
    largest_pval = np.zeros((hor + 1, M))
    # Calculate the bounds for the confidence intervals
    for y in range(M):
        for h in range(hor + 1):
            # Filter grid values for 90% and 68% confidence intervals
            valid_grid_values_90 = grid[p_val[:, h, y] > 0.10]

            valid_grid_values_68 = grid[p_val[:, h, y] > 0.32]

            # Compute bounds for 90% confidence interval
            if len(valid_grid_values_90) > 0:
                lower_bounds_90[h, y] = np.min(valid_grid_values_90)
                upper_bounds_90[h, y] = np.max(valid_grid_values_90)
            else:
                lower_bounds_90[h, y] = np.nan
                upper_bounds_90[h, y] = np.nan
                print('90% Confidence set is empty for y =', y, 'h =', h)

            # Compute bounds for 68% confidence interval
            if len(valid_grid_values_68) > 0:
                lower_bounds_68[h, y] = np.min(valid_grid_values_68)
                upper_bounds_68[h, y] = np.max(valid_grid_values_68)
            else:
                lower_bounds_68[h, y] = np.nan
                upper_bounds_68[h, y] = np.nan
                print('68% Confidence set is empty for y =', y, 'h =', h)

                # Find the central point (maximum p-value)
            max_pval_index = np.argmax(p_val[:, h, y])
            largest_pval[h, y] = grid[max_pval_index]  # we map back our p-val to the grid value

    for y in range(M):
        Structural_Functions.plot_irf_with_confidence_intervals(
            np.arange(hor + 1),
            largest_pval[:, y],
            
            lower_bounds_90[:, y],
            upper_bounds_90[:, y],
            lower_bounds_68[:, y],
            upper_bounds_68[:, y],
            xticks=range(hor + 1),
            central_color='blue',  # Color for the central points
            ci_90_color='red',  # Color for the 90% CI region
            ci_68_color='green',  # Color for the 68% CI region
            title=f'IRF for {Var_data.columns[y]}'
        )
    for y in range(M):
        data_to_plot = [
            largest_pval[:, y],

            upper_bounds_90[:, y],
            lower_bounds_90[:, y],
            upper_bounds_68[:, y],
            lower_bounds_68[:, y],
        ]
        labels = ['Estimate', '90th percentile', '10th percentile', '84th percentile', '16th percentile', ]
        line_styles = ['-', '--', '--', '--', '--']
        line_colors = ['blue', 'red', 'red', 'orange', 'orange']

        Structural_Functions.ploty(data_to_plot, labels, title=f'IRF for {Var_data.columns[y]}', xticks=range(hor + 1),
                                   xlabel="Horizon",
                                   ylabel="Response", line_styles=line_styles, line_colors=line_colors, grid=True)
