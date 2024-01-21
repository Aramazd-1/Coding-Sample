#### Structural Macroeconometrics Final Project ####
import pandas as pd 
import matplotlib.pyplot as plt
from contextlib import contextmanager
import time
import seaborn as sns
import numpy as np
from sklearn import preprocessing # Used to do standardization of variables
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from numpy.linalg import matrix_power
import matlab.engine
from netCDF4 import Dataset
import xarray as xr
from multiprocessing import Pool
import dill as pickle
from numba import njit


#User defined functions 
from Structural_Functions import simulate_series,ploty,compute_irf, ploty_old,analyze_pca,select_blocks, adf_test, add_double_lines_to_latex,generate_blocks, Correlogram,  Companion_Matrix_compute, breusch_test, histogram_Normality_test
from  VAR_estimation import estimate_var_model_const_trend, estimate_var_model_const_trend_ols_boot,VAR_lag_selection

cell_time = {} #Initialize dictionary for storing time taken to run each cell
@contextmanager
def timer(cell_name):
    start_time = time.time()
    yield #Needed to pause execution of timer and then restart it after code is done
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    cell_time[cell_name] = round(elapsed_time,2)
#%% Import and clean climate data
#np.random.seed(0) #Random seed is set for the consistency of bootstrap result across replications, one can deactivate it since results are robust across simulations even without it
path = "D:\Study\Github\Coding-Sample\Python code - Structural Macroeconometrics- Final Project\Data"

climate_data = pd.read_excel(f'{path}\E3CI_data.xlsx', 'Italy_comp_E3CI', index_col=0)
climate_data = climate_data.drop('fire',axis=1).drop('hail',axis=1) #We drop those 2 to conform withne IFAB approach an drop E3CI since it is computed on an older value
climate_data = climate_data.drop('E3CI',axis=1) #We drop the original E3CI since it is computed on an older value
E3CI_mean = climate_data.mean(axis=1)
climate_data.insert(0,"E3CI", E3CI_mean)

climate_data_instrument = pd.read_excel(f'{path}\E3CI_data.xlsx', 'Croatia_comp_E3CI', index_col=0).drop('fire',axis=1).drop('hail',axis=1)
print(f"Mean\n{climate_data.mean(axis=0)}\nStandardDeviation:\n{climate_data.std(axis=0)}") #Manual inspection of data to check if they are standardized or not

#PCA 
PC, Principal_Components = analyze_pca(climate_data.drop('E3CI',axis=1),threshold= 0.90,n_components=4)
loadings = PC.components_
E3CI_PCA = Principal_Components[:,0]
#E3CI_PCA = E3CI_mean #quick command so that if you uncomment you can check what happens when using the mean
E3CI_PCA = pd.DataFrame(E3CI_PCA, index = climate_data.index, columns= ['E3CI_PCA'])
climate_data.insert(0,"E3CI_PCA", E3CI_PCA)
print(f"Mean\n{climate_data['E3CI_PCA'].mean(axis=0)}\nStandardDeviation:\n{climate_data['E3CI_PCA'].std(axis=0)}")
#We now use PCA to construct our instrument
PC_instrument, Principal_Components_instrument = analyze_pca(climate_data_instrument ,threshold= 0.90,n_components=4)
loadings_instrument = PC_instrument.components_
E3CI_PCA_instrument = Principal_Components_instrument[:,0]
#E3CI_PCA_instrument = climate_data_instrument.mean(axis=1) #quick command so that if you uncomment you can check what happens when using the mean
E3CI_PCA_instrument = pd.DataFrame(E3CI_PCA_instrument, index = climate_data_instrument.index, columns= ['E3CI_PCA_instrument'])
climate_data.insert(0,"E3CI_PCA_instrument", E3CI_PCA_instrument)

#Housekeeping and cleaning
climate_dataframes = {}
climate_dataframes.update({ 'climate_data': climate_data,'climate_data_instrument': climate_data_instrument, 'loadings': [loadings, loadings_instrument]})
del climate_data, climate_data_instrument, E3CI_mean, PC, PC_instrument, Principal_Components, Principal_Components_instrument, loadings, loadings_instrument
#%% Import and clean data macro ###
# note we are trying to make sure that all our data is in dataframes here!

start_date = pd.to_datetime('1997-07-01')
end_date = pd.to_datetime('2023-06-30') 
dates = {}
dates.update({'start_date': start_date,'end_date': end_date })

 
df_macro_eurostat = pd.read_excel(f"{path}\Working Data_EUROSTAT.xlsx" , index_col=0).dropna().drop('Euro_Short_Rate', axis =1).drop('ALL_HICP', axis =1).astype(float)
df_macro_eurostat.index = pd.to_datetime(df_macro_eurostat.index, format = '%Y-%m')
decomposition = sm.tsa.seasonal_decompose(df_macro_eurostat['HICP_FE'], model='multiplicative', period=12).seasonal
df_macro_eurostat['HICP_FE'] = df_macro_eurostat['HICP_FE'].div(decomposition, axis=0) 
df_macro_eurostat = np.log(df_macro_eurostat).diff(periods=1).dropna()*100 #year over year growth of non food and energy inflation
df_macro_eurostat = df_macro_eurostat[(df_macro_eurostat.index >= start_date) & (df_macro_eurostat.index <= end_date)] 

#♥loads energy_uncertainty_italy, macro unc and iprod
df_macro_else = pd.read_excel(f"{path}\Working Data_Else.xlsx", index_col=0).drop('energy_uncertainty', axis=1)
df_macro_else.index = pd.to_datetime(df_macro_else.index, format = '%Y-%m')
df_macro_else = df_macro_else[(df_macro_else.index >= start_date) & (df_macro_else.index <= end_date)]#.drop('energy_uncertainty',axis=1) 


df_ECB = pd.read_excel(f"{path}\Working Data_ECB.xlsx",'B', index_col=0)
df_ECB.index = pd.to_datetime(df_ECB.index, format = '%Y-%m')
decomposition = sm.tsa.seasonal_decompose(df_ECB['HICP'], model='multiplicative', period=12)
df_ECB['HICP'] = (df_ECB['HICP'].div(decomposition.seasonal, axis=0)).div(decomposition.trend, axis=0) #we also detrend this inflation
df_ECB.loc[:, df_ECB.columns != '3monthrate'] = df_ECB.loc[:, df_ECB.columns != '3monthrate'].apply(lambda x: (np.log(x).diff(periods=12) ) *100 ) #Difference over precedent year of GDP index and HICP
df_ECB['3monthrate'] = df_ECB['3monthrate'].diff(periods=1) #first difference of 3 month rate
df_ECB = df_ECB[(df_ECB.index >= start_date) & (df_ECB.index <= end_date)]
df_ECB.index = df_macro_else.index 



heatgas_prices = np.log(pd.read_excel(f'{path}\heatgas_prices.xlsx', index_col=0)).diff(periods=1).dropna()
heatgas_prices.index = pd.to_datetime(heatgas_prices.index, format = '%Y-%m')
heatgas_prices = heatgas_prices[(heatgas_prices.index >= start_date) & (heatgas_prices.index <= end_date)]

Var_data = pd.concat([df_macro_eurostat, df_macro_else,df_ECB], axis=1).dropna()#.drop('GDP_Index',axis=1)
E3CI_PCA = E3CI_PCA[(E3CI_PCA.index >= start_date) & (E3CI_PCA.index <= end_date)]
E3CI_PCA_instrument = E3CI_PCA_instrument[(E3CI_PCA_instrument.index >= start_date) & (E3CI_PCA_instrument.index <= end_date)]
Var_data.insert(0,"E3CI_PCA", E3CI_PCA), Var_data.insert(1, 'net_gasprices', heatgas_prices)
Var_data = Var_data.drop('Uncertainty_Italy_News', axis=1).drop('HICP_FE', axis= 1).drop('Iprod_Italy_Year_on_year_growth', axis=1)
Var_data.to_excel(excel_writer='D:\Study\Github\Coding-Sample\Python code - Structural Macroeconometrics- Final Project\Data\Var_data.xlsx', index=False)

#Housekeeping and cleaning
del df_macro_eurostat, df_macro_else, start_date, end_date, heatgas_prices, decomposition, df_ECB
#%% Preliminary Plotting:
for _ in climate_dataframes['climate_data'].columns: ploty_old(climate_dataframes['climate_data'][f"{_}"], title = f"{_}") # Plotting loop
for _ in Var_data.columns: ploty_old(Var_data[f"{_}"], title = f"{_}")         # Plotting loop
#%%
with timer('Var cell'):
    adf_test(Var_data) #We run an ADF test on the variables in our dataset to check for their stationarity 
    eng = matlab.engine.start_matlab() #The bulk of the time here is due to matlab boot time
    criteria_df = VAR_lag_selection(Var_data,eng, max_p=4)
    Non_ar_params, Sigma_u, AR_matrices, Residuals = estimate_var_model_const_trend(Var_data,eng, p=2) #We use MATLAB to estimate a VAR
    
    print(add_double_lines_to_latex(criteria_df.to_latex(index=True, escape=False, column_format='c c c c'))) #To manually export the table to latex using console, file.write instead of print if you prefer
#%% We now proceed to estimate the SVAR model
M = AR_matrices["AR1"].shape[0] # Number of variables
T = len(Var_data) - len(AR_matrices) #Effective sample
p = len(AR_matrices) #number of lags
hor = 24 #The horizon of our IRFs
k = M*2+ M*M*p #Number of estimated parameters (m*2 constants, trend+intercept) and therest are the AR matrices M*M for each p

# sea_level_data = pd.read_excel(f'{path}\Global _Sea Level.xlsx',index_col=0) #Load the instrument
# sea_level_data.index = pd.to_datetime(sea_level_data.index)
# instrument = sea_level_data.resample('M').mean() # Resample the data to monthly frequency and calculate the mean
# instrument = instrument.diff(periods=1).dropna()

instrument = E3CI_PCA_instrument

if len(Var_data) == len(instrument): print('Var dataset has same length as instrument')
else: raise ValueError(f"Var dataset doesn't have the same length as the instrument, please revise dates by {len(instrument)-len(Var_data)}")

print(f'std dev:{np.std(instrument)} ')
adf_test(instrument)


 
Companion_Matrix =  Companion_Matrix_compute(AR_matrices, M)
eigenvalues = np.linalg.eigvals(Companion_Matrix) #We can manually inspect the presence of a unitary root


phi_hat = 1/(T-p) * np.dot(Residuals.T, instrument.iloc[p:]) # We compute the sample covariance between the instrument and the Residuals
rhosqu_hat = phi_hat.T@ np.linalg.inv(Sigma_u) @ phi_hat

if rhosqu_hat < 0:
    raise ValueError("Negative rhosqu_hat") #Simply a sanity check for internal troubleshooting, should never be actually flagged in any relevant case
rho_hat_std = (rhosqu_hat**(1/2))
corr_instr= rho_hat_std/ (np.std(instrument.iloc[p:]).item())
k_hat = rhosqu_hat**(-1/2)*phi_hat

if rhosqu_hat**(-1/2) < 0:
    raise ValueError("k_hat negative !") #Simply a sanity check for internal troubleshooting, should never be actually flagged in any relevant case
if p>1 :R_selection= np.hstack([np.eye(M), np.zeros((M, M*(p-1)))])
else: R_selection = np.eye(M)
IRF_hat_matrix = []
for h in range (0, hor+1):
    IRF_hat_h =  (R_selection @ matrix_power(Companion_Matrix, h) @R_selection.T @ k_hat).T
    IRF_hat_matrix.append(IRF_hat_h)

IRF_hat_matrix = pd.DataFrame(np.vstack(IRF_hat_matrix), index = range(0,hor+1), columns=Var_data.columns.tolist())
for _ in IRF_hat_matrix.columns: ploty_old(IRF_hat_matrix[f'{_}'], title=f'{_}')
#%% Bootstrap (RECALL TO MAKE THIS INTO A FUNCTION AND IMPROVE SPEED DRASTICALLY)


#Some exploratory testing, there is more than what presented, we did more than needed to explore our data
#for _ in pd.DataFrame(Residuals).columns : print(f'Normality test {_}'), histogram_Normality_test(pd.DataFrame(Residuals), series = 'Residuals',column = _, bins = 50, frequency = 'monthly')
#for _ in pd.DataFrame(Residuals).columns : print(f'Box test {_+1}'), print(sm.stats.acorr_ljungbox(pd.DataFrame(Residuals[:,_]), lags = 10, boxpierce = True, return_df=True))
# for _ in pd.DataFrame(Residuals).columns : Correlogram(pd.DataFrame(Residuals[:,_]), lower_bound =-0.4, upper_bound = 0.4, frequency = 'Monthly', titlez = f'Residual {_+1}')
# for _ in pd.DataFrame(Residuals).columns : Correlogram(pd.DataFrame(Residuals[:,_]**2), lower_bound =-0.4, upper_bound = 0.4, frequency = 'Monthly', titlez = f'Squared Residual {_+1}')
# for _ in pd.DataFrame(Residuals).columns : ploty(data= pd.DataFrame(Residuals[:,_]))
# adf_test(pd.DataFrame(Residuals))
#To improve the boot loop speed one could run in parallel some estimation and then vertically stack the outcome matrices 
with timer('Boot cell'):
    #eng = matlab.engine.start_matlab()  #Uncomment to the left if you need to start again matlab engine
    
    #Initializing external parameters
    block_length=12
    Residuals_expanded = np.hstack((Residuals,instrument[p:])) #we stack all residuals together to jointly boot them
    blocks = generate_blocks(Residuals_expanded, block_length = block_length)
    nboot = 10 #Number of boot iterations
    IRF_matrices_boot = np.zeros((nboot, hor+1, M)) #3D array to store Bootstrap results
    phi_hat_array_boot = np.zeros((nboot,M))
    k_hat_array_boot = np.zeros((nboot,M))
    
    for b in range(0, nboot):
        if (b+1) %10==0: print(f'botstrap iteration: {b+1}')
        Sample_boot = np.zeros((T+p, M))  #We Initialize the sample
        Sample_boot[:p, :] = Var_data.iloc[:p].to_numpy() #makes first p values equal to the original sample
        
        Residuals_boot = select_blocks(blocks, int(T/block_length)+1)[:T,:] #Randomly samples block with replacement
        Residuals_boot = Residuals_boot - np.mean(Residuals_boot, axis=0, keepdims=True) #We center residuals
        instrument_boot = Residuals_boot[:,M]  #Selects the last column to be the Residuals of the instrument
        Residuals_boot = Residuals_boot[:,:M]
        
        
        for i in range(p,T): Sample_boot[i] = Non_ar_params.iloc[:,0].T + (Non_ar_params.iloc[:,1].T)*i + sum(AR_matrices[f'AR{j}'] @ Sample_boot[i-j].T for j in range(1, p + 1) if f'AR{j}' in AR_matrices) + Residuals_boot[i-2,:]           
        
        #Estimation of the model and IRF
        #Non_ar_params_boot, Sigma_u_boot, AR_matrices_boot, Residuals_boot = estimate_var_model_const_trend(Sample_boot, eng, p=p, noprint=True)
        Sigma_u_boot, AR_matrices_boot, Residuals_boot = estimate_var_model_const_trend_ols_boot(Sample_boot, p=p, noprint=True)
        phi_hat_boot = 1/(T-p) * np.dot(Residuals_boot.T, instrument_boot) # We compute the sample covariance between the instrument and the Residual
        rho_squared_hat_boot = phi_hat_boot.T@ np.linalg.inv(Sigma_u_boot) @ phi_hat_boot
        if rho_squared_hat_boot**(-1/2) < 0:
            raise ValueError("Negative rhosqu_hat")
        k_hat_boot = rho_squared_hat_boot**(-1/2)*phi_hat_boot
        Companion_Matrix_boot =  Companion_Matrix_compute(AR_matrices_boot, M)
        

        
        IRF_hat_matrix_boot = np.zeros((hor+1, M))
        for h in range (0, hor+1):
            IRF_hat_matrix_boot[h, :] = (R_selection @ matrix_power(Companion_Matrix_boot, h) @ R_selection.T @ k_hat_boot).T

        IRF_matrices_boot[b, :, :] = np.vstack(IRF_hat_matrix_boot)
    #The loop basically ends whilst returning the IRF matrices for bootstrapping!
    avg_IRF_hat_matrix_boot = np.zeros((hor+1, M)) #Bootstrap mean to check for bias
    top_IRF_hat_matrix_boot =np.zeros((hor+1, M))  #95 percent confidence interval
    top68_IRF_hat_matrix_boot =np.zeros((hor+1, M))
    bot68_IRF_hat_matrix_boot =np.zeros((hor+1, M))
    top90_IRF_hat_matrix_boot =np.zeros((hor+1, M))
    bot10_IRF_hat_matrix_boot =np.zeros((hor+1, M))
    bot_IRF_hat_matrix_boot = np.zeros((hor+1, M)) #5 percent confidence interval
    for h in range(0,hor+1):
            avg_IRF_hat_matrix_boot[h, :] = np.mean(IRF_matrices_boot[:, h, :], axis=0)  # Averaging the h row across all nboot matrices
            top_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 95, axis=0)
            bot_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 5, axis=0)
            top68_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 84, axis=0)
            bot68_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 16, axis=0)
            top90_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 90, axis=0)
            bot10_IRF_hat_matrix_boot[h, :] = np.percentile(IRF_matrices_boot[:, h, :], 10, axis=0)
            
           
            
    avg_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(avg_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    top_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    bot_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    top68_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top68_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    bot68_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot68_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    top90_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top90_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    bot10_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot10_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())

#%% Plotting IRFs
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
    labels = ['Estimate','Mean', '95th Percentile', '5th Percentile','84th percentile','16th percentile','90th percentile', '10th percentile']
    line_styles = ['-','-', '--', '--','--','--','--','--']
    line_colors = ['blue','green', 'red', 'red', 'orange','orange','purple','purple']

    ploty(data_to_plot, labels, title=f"IRF for {_}",xticks=range(hor+1), xlabel="Horizon", ylabel="Response", line_styles=line_styles, line_colors=line_colors, grid=True)
#%%
for _ in range(0,M):
    data_to_plot_2 = [
        Residuals_boot[:,_],
        Residuals[:,_]
    ]
    labels = ['boot','true' ]
    line_styles = ['-', '--']
    line_colors = ['red', 'blue']

    ploty(data_to_plot_2, labels, title=f"IRF for {_}", xlabel="Horizon", ylabel="Response", line_styles=line_styles, line_colors=line_colors, grid=True)    
#%% New boot for faster
    
    

        
        