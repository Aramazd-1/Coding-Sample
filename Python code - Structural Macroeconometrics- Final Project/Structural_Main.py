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

#User defined functions 
from Structural_Functions import simulate_series,ploty, ploty_old,analyze_pca,select_blocks, adf_test, add_double_lines_to_latex,generate_blocks, Correlogram,  Companion_Matrix_compute, breusch_test, histogram_Normality_test
from  VAR_estimation import estimate_var_model_const_trend, VAR_lag_selection

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
climate_data = climate_data.drop('E3CI',axis=1) #We drop E3CI since it is computed on an older value
E3CI_mean = climate_data.mean(axis=1)
climate_data.insert(0,"E3CI", E3CI_mean)

print(f"Mean\n{climate_data.mean(axis=0)}\nStandardDeviation:\n{climate_data.std(axis=0)}") #Manual inspection of data to check if they are standardized or not

#%% PCA 
PC, Principal_Components = analyze_pca(climate_data.drop('E3CI',axis=1),threshold= 0.90,n_components=4)
loadings = PC.components_
E3CI_PCA = Principal_Components[:,0]
climate_data.insert(0,"E3CI_PCA", E3CI_PCA)
print(f"Mean\n{climate_data['E3CI_PCA'].mean(axis=0)}\nStandardDeviation:\n{climate_data['E3CI_PCA'].std(axis=0)}")

#%% Import and clean data macro
start_date = pd.to_datetime('1997-01-01')
end_date = pd.to_datetime('2022-08-31')
#☼ filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)] to be implemented
df_macro_eurostat = pd.read_excel(f"{path}\Working Data_EUROSTAT.xlsx",'B' , index_col=0).dropna()
df_macro_else = pd.read_excel(f"{path}\Working Data_Else.xlsx", index_col=0).dropna()
#DF_INFLATION = pd.read_excel(f"{path}\Inflation_Italy_grwoth.xlsx", index_col=0).dropna()
#DF_INFLATION = DF_INFLATION.diff(periods=1).dropna()
Var_data = pd.concat([df_macro_eurostat, df_macro_else], axis=1).drop('HICP_Food_Energy', axis  =1).drop('HICP_NOT_FE', axis  =1)
Var_data.insert(0,"E3CI_PCA", E3CI_PCA[192:])
#Var_data.insert(0,"E3CI_Mean", E3CI_mean) #To compare with the performance you would get by using the mean
Var_data.index = pd.to_datetime(Var_data.index, format='%Y-%m') #This ensures Python correctly reads our dates as such
Var_data = Var_data.iloc[:-10]


#%% Preliminary Plotting:
for _ in climate_data.columns: ploty_old(climate_data[f"{_}"], title = f"{_}") # Plotting loop
for _ in Var_data.columns: ploty_old(Var_data[f"{_}"], title = f"{_}")         # Plotting loop

#%% VAR here we will use the matlab api to estimate a Var model using varm and estimate
with timer('Var cell'):
    adf_test(Var_data) #We run an ADF test on the variables in our dataset to check for their stationarity 
    eng = matlab.engine.start_matlab() #The bulk of the time here is due to matlab boot time
    
    criteria_df = VAR_lag_selection(Var_data,eng, max_p=7)
    Non_ar_params, logLikVAR, Sigma_u, AR_matrices, Residuals = estimate_var_model_const_trend(Var_data,eng, p=2) #We use MATLAB to estimate a VAR
    
    print(add_double_lines_to_latex(criteria_df.to_latex(index=True, escape=False, column_format='c c c c'))) #To manually export the table to latex using console, file.write instead of print if you prefer
    

#%% We now proceed to estimate the SVAR model
M = AR_matrices["AR1"].shape[0] # Number of variables
T = len(Var_data) - len(AR_matrices) #Effective sample
p= len(AR_matrices) #number of lags
hor = 24 #The horizon of our IRFs
k = M*2+ M*M*p #Number of estimated parameters (m*2 constants, trend+intercept) and therest are the AR matrices M*M for each p

sea_level_data = pd.read_excel(f'{path}\Global _Sea Level.xlsx',index_col=0) #Load the instrument
sea_level_data.index = pd.to_datetime(sea_level_data.index)
instrument = sea_level_data.resample('M').mean() # Resample the data to monthly frequency and calculate the mean
#○instrument = pd.read_excel(f'{path}\Sea Surface Temperature.xlsx',index_col=0) #Load the instrument sea temperature
#instrument = instrument.iloc[48:]
#instrument = instrument.iloc[3:]
adf_test(instrument)
instrument = instrument.diff(periods=1).dropna()
instrument = instrument[len(AR_matrices):] 
adf_test(instrument)


Companion_Matrix =  Companion_Matrix_compute(AR_matrices, M)
eigenvalues = np.linalg.eigvals(Companion_Matrix) #We can manually inspect the presence of a unitary root


phi_hat = 1/(T-k) * np.dot(Residuals.T, instrument) # We compute the sample covariance between the instrument and the Residuals


rhosqu_hat = phi_hat.T@ np.linalg.inv(Sigma_u) @ phi_hat
abs(rhosqu_hat**(-1/2))
if rhosqu_hat < 0:
    raise ValueError("Negative rhosqu_hat") #Simply a sanity check for internal troubleshooting, should never be actually flagged in any relevant case
rho_hat_std = (rhosqu_hat**(1/2))
corr_instr= rho_hat_std/ (np.std(instrument).item())
k_hat = rhosqu_hat**(-1/2)*phi_hat

if rhosqu_hat**(-1/2) < 0:
    raise ValueError("k_hat negative !") #Simply a sanity check for internal troubleshooting, should never be actually flagged in any relevant case
R_selection= np.hstack([np.eye(M), np.zeros((M, M))])
IRF_hat_matrix = []
for h in range (0, hor+1):
    IRF_hat_h =  (R_selection @ matrix_power(Companion_Matrix, h) @R_selection.T @ k_hat).T
    IRF_hat_matrix.append(IRF_hat_h)

IRF_hat_matrix = pd.DataFrame(np.vstack(IRF_hat_matrix), index = range(0,hor+1), columns=Var_data.columns.tolist())

#%% Investigation of the autoregressive behaviour of the Instrument

criteria_df = VAR_lag_selection(instrument,eng, max_p=12)
#%% Bootstrap (RECALL TO MAKE THIS INTO A FUNCTION AND IMPROVE SPEED DRASTICALLY)


#Some testing, there is more than what presented, we did more than needed to explore our data
# for _ in pd.DataFrame(Residuals).columns : print(f'Normality test {_}'), histogram_Normality_test(pd.DataFrame(Residuals), series = 'Residuals',column = _, bins = 50, frequency = 'monthly')
# for _ in pd.DataFrame(Residuals).columns : print(f'Box test {_+1}'), print(sm.stats.acorr_ljungbox(pd.DataFrame(Residuals[:,_]), lags = 10, boxpierce = True, return_df=True))
# for _ in pd.DataFrame(Residuals).columns : Correlogram(pd.DataFrame(Residuals[:,_]), lower_bound =-0.4, upper_bound = 0.4, frequency = 'Monthly', titlez = f'Residual {_+1}')
# for _ in pd.DataFrame(Residuals).columns : Correlogram(pd.DataFrame(Residuals[:,_]**2), lower_bound =-0.4, upper_bound = 0.4, frequency = 'Monthly', titlez = f'Squared Residual {_+1}')
# for _ in pd.DataFrame(Residuals).columns : ploty(data= pd.DataFrame(Residuals[:,_]))
# adf_test(pd.DataFrame(Residuals))
#To improve the boot loop speed one could run in parallel some estimation and then vertically stack the outcome matrices 
with timer('Boot cell'):
    eng = matlab.engine.start_matlab() #The bulk of the time here is due to matlab boot time
    #Initializing external parameters
    block_length=9 #Manually inspect if T is a multiple of block_length (currently handling of)
    p_instr = 2 #Number of lags describing the DGP of the instrument 
    blocks = generate_blocks(Residuals, block_length = block_length)
    nboot = 999 #Number of boot iterations
    IRF_matrices_boot = np.zeros((nboot, hor+1, M)) #3D array to store Bootstrap results
    phi_hat_array_boot = np.zeros((nboot,M))
    for b in range(0, nboot):
        print(f'Botstrap iteration: {b+1}')
        Sample_boot = np.zeros((T+p, M))  #We Initialize the sample
        #Instrument_boot = np.zeros((T+p_instr,M))
        Sample_boot[0,:] = Var_data.iloc[0]
        Sample_boot[1,:] = Var_data.iloc[1]
        Residuals_boot = select_blocks(blocks, int(T/block_length)) #Randomly samples block with replacement
        for i in range(p,T+p): #Loop generating the Sample
            Sample_boot[i] = Non_ar_params.iloc[:,0].T + (Non_ar_params.iloc[:,1].T)*i + AR_matrices['AR1'] @ Sample_boot[i-1] + AR_matrices['AR2'] @ Sample_boot[i-2].T + Residuals_boot[i-2,:]
       
        #Estimation of the model and IRF
        Non_ar_params_boot, logLikVAR , Sigma_u_boot, AR_matrices_boot, Residuals_boot = estimate_var_model_const_trend(Sample_boot, eng, p=2, noprint=True)
        #phi_hat_boot = 1/(T-1) * np.dot(Residuals_boot.T, instrument) # We compute the sample covariance between the instrument and the Residuals
        phi_hat_boot = phi_hat
        rho_squared_hat_boot = phi_hat_boot.T@ np.linalg.inv(Sigma_u_boot) @ phi_hat_boot
        if rho_squared_hat_boot**(-1/2) < 0:
            raise ValueError("Negative rhosqu_hat")
        k_hat_boot = rho_squared_hat_boot**(-1/2)*phi_hat_boot
        Instrument_boot = instrument
        Companion_Matrix_boot =  Companion_Matrix_compute(AR_matrices_boot, M)
        
        IRF_hat_matrix_boot = []
        for h in range (0, hor+1):
            IRF_hat_h =  (R_selection @ matrix_power(Companion_Matrix_boot, h) @R_selection.T @ k_hat_boot).T
            IRF_hat_matrix_boot.append(IRF_hat_h)
        #phi_hat_array_boot[b,:]= phi_hat_boot.T
        IRF_matrices_boot[b, :, :] = np.vstack(IRF_hat_matrix_boot)
    #The loop basically ends whilst returning the IRF matrices for bootstrapping!
    avg_IRF_hat_matrix_boot = [] #Bootstrap mean to check for bias
    top_IRF_hat_matrix_boot =[]  #95 percent confidence interval
    bot_IRF_hat_matrix_boot = [] #5 percent confidence interval
    for h in range(0,hor+1):
            average_IRF_h = np.mean(IRF_matrices_boot[:, h, :], axis=0)  # Averaging the h row across all nboot matrices
            percentile_95_h = np.percentile(IRF_matrices_boot[:, h, :], 95, axis=0)
            percentile_5_h = np.percentile(IRF_matrices_boot[:, h, :], 5, axis=0)
            avg_IRF_hat_matrix_boot.append(average_IRF_h)
            top_IRF_hat_matrix_boot.append(percentile_95_h)
            bot_IRF_hat_matrix_boot.append(percentile_5_h)
            
    avg_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(avg_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    top_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(top_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())
    bot_IRF_hat_matrix_boot = pd.DataFrame(np.vstack(bot_IRF_hat_matrix_boot), index = range(0,hor+1), columns=Var_data.columns.tolist())

    
#%% Plotting IRFs
for _ in Var_data.columns:
    data_to_plot = [
        IRF_hat_matrix[_],
        avg_IRF_hat_matrix_boot[_],
        top_IRF_hat_matrix_boot[_],
        bot_IRF_hat_matrix_boot[_]
    ]
    labels = ['Estimate','Mean', '95th Percentile', '5th Percentile']
    line_styles = ['-','-', '--', '--']
    line_colors = ['blue','green', 'red', 'red']

    ploty(data_to_plot, labels, title=f"IRF for {_}", xlabel="Horizon", ylabel="Response", line_styles=line_styles, line_colors=line_colors, grid=True)
for _ in range(0,M):
    data_to_plot_2 = [
        Residuals_boot[:,_],
        Residuals[:,_]
    ]
    labels = ['boot','true' ]
    line_styles = ['-', '--']
    line_colors = ['red', 'blue']

    ploty(data_to_plot_2, labels, title=f"IRF for {_}", xlabel="Horizon", ylabel="Response", line_styles=line_styles, line_colors=line_colors, grid=True)    
    
    
    
    
    
    
    

        
        