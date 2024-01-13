#### Structural Macroeconometrics Final Project ####
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing # Used to do standardization of variables
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from numpy.linalg import matrix_power

#User defined functions 
from Structural_Functions import simulate_series,ploty, analyze_pca,select_blocks, adf_test, add_double_lines_to_latex,generate_blocks, Correlogram,  Companion_Matrix_compute, breusch_test, histogram_Normality_test
from  VAR_estimation import estimate_var_model_const_trend, VAR_lag_selection

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
df_macro_eurostat = pd.read_excel(f"{path}\Working Data_EUROSTAT.xlsx",'B' , index_col=0).dropna()
df_macro_else = pd.read_excel(f"{path}\Working Data_Else.xlsx", index_col=0).dropna()

Var_data = pd.concat([df_macro_eurostat, df_macro_else], axis=1)
Var_data.insert(0,"E3CI_PCA", E3CI_PCA[192:])
#Var_data.insert(0,"E3CI_Mean", E3CI_mean) #To compare with the performance you would get by using the mean
Var_data.index = pd.to_datetime(Var_data.index, format='%Y-%m') #This ensures Python correctly reads our dates as such
Var_data = Var_data.iloc[:-10]

#%% Preliminary Plotting:
for _ in climate_data.columns: ploty(climate_data[f"{_}"], title = f"{_}") # Plotting loop
for _ in Var_data.columns: ploty(Var_data[f"{_}"], title = f"{_}")         # Plotting loop

#%% VAR here we will use the matlab api to estimate a Var model using varm and est imate
adf_test(Var_data) #We run an ADF test on the variables in our dataset ot check for their stationarity


criteria_df = VAR_lag_selection(Var_data, max_p=7)
Non_ar_params, logLikVAR, Sigma_u, AR_matrices, Residuals = estimate_var_model_const_trend(Var_data, p=2) #We use MATLAB to estimate a VAR

print(add_double_lines_to_latex(criteria_df.to_latex(index=True, escape=False, column_format='c c c c'))) #To manually export the table to latex using console, file.write instead of print if you prefer


#%% We now proceed to estimate the SVAR model
M = AR_matrices["AR1"].shape[0] # Number of variables
T = len(Var_data) - len(AR_matrices) #Effective sample
p= len(AR_matrices) #number of lags

sea_level_data = pd.read_excel(f'{path}\Global _Sea Level.xlsx',index_col=0) #Load the instrument
sea_level_data.index = pd.to_datetime(sea_level_data.index)
instrument_sea_level = sea_level_data.resample('M').mean()# Resample the data to monthly frequency and calculate the mean
instrument_sea_level = instrument_sea_level[len(AR_matrices):]



Companion_Matrix =  Companion_Matrix_compute(AR_matrices, M)
eigenvalues = np.linalg.eigvals(Companion_Matrix)


phi_hat = 1/(T-1) * np.dot(Residuals.T, instrument_sea_level) # We compute the sample covariance between the instrument and the Residuals
rho_squared_hat = phi_hat.T@ np.linalg.inv(Sigma_u) @ phi_hat
k_hat = rho_squared_hat**(-1/2)*phi_hat
R_selection= np.hstack([np.eye(M), np.zeros((M, M))])
IRF_hat_results = []
for h in range (1, 100):
    IRF_hat_h =  (R_selection @ matrix_power(Companion_Matrix, h) @R_selection.T @ k_hat).T
    IRF_hat_results.append(IRF_hat_h)

IRF_hat_matrix = pd.DataFrame(np.vstack(IRF_hat_results), index = range(1,100), columns=Var_data.columns.tolist())
for _ in IRF_hat_matrix.columns: ploty(IRF_hat_matrix[f"{_}"], title = f"{_}") # Plotting loop, to be substit nuted when plotting bands are available


#%% Bootstrap
#Preliminary testing, there is more than what presented, we did more than needed to explore our data
# for _ in pd.DataFrame(Residuals).columns : print(f'Normality test {_}'), histogram_Normality_test(pd.DataFrame(Residuals), series = 'Residuals',column = _, bins = 50, frequency = 'monthly')
# for _ in pd.DataFrame(Residuals).columns : print(f'Box test {_+1}'), print(sm.stats.acorr_ljungbox(pd.DataFrame(Residuals[:,_]), lags = 10, boxpierce = True, return_df=True))
# for _ in pd.DataFrame(Residuals).columns : Correlogram(pd.DataFrame(Residuals[:,_]), lower_bound =-0.4, upper_bound = 0.4, frequency = 'Monthly', titlez = f'Residual {_+1}')
# for _ in pd.DataFrame(Residuals).columns : Correlogram(pd.DataFrame(Residuals[:,_]**2), lower_bound =-0.4, upper_bound = 0.4, frequency = 'Monthly', titlez = f'Squared Residual {_+1}')
# for _ in pd.DataFrame(Residuals).columns : ploty(data= pd.DataFrame(Residuals[:,_]))
# adf_test(pd.DataFrame(Residuals))
block_length=6 #Manually inspect if T is a multiple of block_length (currently handling of)
blocks = generate_blocks(Residuals, block_length = block_length)
for b in range(1, 5):
    Sample_boot = np.zeros((T+p, M))  #We Initialize the sample
    Sample_boot[0,:] = Var_data.iloc[0]
    Sample_boot[1,:] = Var_data.iloc[1]
    Residuals_boot = select_blocks(blocks, int(T/block_length))
    for i in range(p,T+p):
        Sample_boot[i] = Non_ar_params.iloc[:,0].T + (Non_ar_params.iloc[:,1].T)*i + AR_matrices['AR1'] @ Sample_boot[i-1] + AR_matrices['AR2'] @ Sample_boot[i-2].T + Residuals_boot[i-2,:]
        #Regenerate Instrument
Residuals_Boot = pd.DataFrame(Residuals_boot, columns = Var_data.columns.tolist())
for _ in Residuals_Boot.columns: ploty(Residuals_Boot[f"{_}"], title = f"{_}")
        
        