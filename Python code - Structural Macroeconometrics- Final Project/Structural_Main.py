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

#User defined functions and parameters
from Structural_Functions import simulate_series,ploty, analyze_pca, adf_test
from  VAR_estimation import estimate_var_model
#from Parameters import kwargs_list #I defined them for a student t, but one ought to define its own list of dictionaries with parameters to pass.
#%% Import and clean data climate
np.random.seed(0) #Random seed is set for the consistency of bootstrap result across replications, one can deactivate it since results are robust across simulations even without it
path = "D:\Study\Github\Coding-Sample\Python code - Structural Macroeconometrics- Final Project\Data"

climate_data = pd.read_excel(f'{path}\E3CI_data.xlsx', 'Italy_comp_E3CI', index_col=0)
climate_data = climate_data.drop('hail',axis=1)
climate_data = climate_data.drop('fire',axis=1) #We drop those to conform withne IFAB approach 
climate_data = climate_data.drop('E3CI',axis=1) #We drop it as it is computed on an older value
E3CI_mean =climate_data.mean(axis=1)
climate_data.insert(0,"E3CI", E3CI_mean)


print(f"Mean\n{climate_data.mean(axis=0)}\nStandardDeviation:\n{climate_data.std(axis=0)}") #Manual inspection of data to check if they are standardized or not
#%% PCA 
PC, Principal_Components = analyze_pca(climate_data.drop('E3CI',axis=1),threshold= 0.90,n_components=4)
loadings = PC.components_
E3CI_PCA = Principal_Components[:,0]
#%% Import and clean data macro
df_macro_eurostat = pd.read_excel(f"{path}\Working Data_EUROSTAT.xlsx",'B' , index_col=0).dropna()
df_macro_else = pd.read_excel(f"{path}\Working Data_Else.xlsx", index_col=0).dropna()

Var_data = pd.concat([df_macro_eurostat, df_macro_else], axis=1)
Var_data.insert(0,"E3CI_PCA", E3CI_PCA[192:])
Var_data.index = pd.to_datetime(Var_data.index, format='%Y-%m') #This ensures Python correctly reads our dates as such
#%%
climate_data.insert(0,"E3CI_PCA", E3CI_PCA)
print(f"Mean\n{climate_data['E3CI_PCA'].mean(axis=0)}\nStandardDeviation:\n{climate_data['E3CI_PCA'].std(axis=0)}")
#%% Preliminary Plotting:
for _ in climate_data.columns: ploty(climate_data[f"{_}"], title = f"{_}") # Plotting loop
for _ in Var_data.columns: ploty(Var_data[f"{_}"], title = f"{_}") # Plotting loop
#%% VAR here we will use the matlab api to estimate a Var model using varm and estimate
adf_test(Var_data) #We run an ADF test on the variables in our dataset ot check for their stationarity
Const, Trend, logLikVAR, Sigma_u, a = estimate_var_model(Var_data, p=4)
