#%% Preliminaries and Importing
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
from Functions.EstimationOLS import ols
from Functions.Simulation import simulate_data
from Functions.Simulation import simulate_rel
from Functions.Kfold import k_fold_cv
from Functions.EstimationRidge import Ridge_a
from Functions.EstimationLasso import Lasso_a
from Functions.PlotErrors import plot_errors
from functools import partial
import matplotlib.pyplot as plt
cell_time = {} #Initialize dictionary for storing time taken to run each cell
RidgeTestMSE_alpha = []
LassoTestMSE_alpha = []
@contextmanager
def timer(cell_name):
    start_time = time.time()
    yield #Needed to pause execution of timer and then restart it after code is done
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    cell_time[cell_name] = round(elapsed_time,2)
#%% Simulation Cell
with timer("Simulation Cell"): #starts time count
 np.random.seed(100)
 True_Parameters = simulate_rel(600,600,3) #Simulates true parameters from huge distribution
 data = {}  #create a dictionary
 data['Simulated_data'] = simulate_data(2400,600,600,4, True_Parameters['true_betas'],True_Parameters['error_betas'],True_Parameters['sd_values'],True_Parameters['sd_values_rnd'])  # inserts values in the dictionary with key "simulated data"

#%% OLS Cell
with timer("OLS Cell"): #starts time count 
 best_parameters_OLS = {}
 best_parameters_OLS['train_errors_OLS'], best_parameters_OLS['test_errors_OLS'], best_parameters_OLS['best_ols_params'],best_parameters_OLS['best_ols_model'] = k_fold_cv(data['Simulated_data'], ols, k=5, dependent_var='y')
 plot_errors(best_parameters_OLS['train_errors_OLS'],best_parameters_OLS['test_errors_OLS'], 5 , 'OLS')
 print(" MSE:" , np.mean(best_parameters_OLS['test_errors_OLS']))
#%% Ridge Cell
with timer("Ridge Cell"): #starts time count

    alphas_Ridge = np.linspace(10000,16000,41) #generates 41 values from 10000 t0 16000
    best_parameters_Ridge = {  
    'best_test_mse_Ridge': float('inf'),
     }
     #all_test_errors = {}  # Dictionary to store test errors for each alpha

    def ridge_wrapper(alpha):
        def ridge_fixed_alpha(data, dependent_var='y'):
            return Ridge_a(data, alpha=alpha, dependent_var=dependent_var)
        return ridge_fixed_alpha
    
    def lasso_wrapper(alpha):
        def lasso_fixed_alpha(data, dependent_var='y'):
            return Lasso_a(data, alpha=alpha, dependent_var=dependent_var)
        return lasso_fixed_alpha
    
    #reg is the dictionary where we store data
    def alphaloop (alpha, reg, regstr,wrapper):
        ridge_fixed = wrapper(alpha)
        train_errors,test_errors, model_params, _ = k_fold_cv(data['Simulated_data'], ridge_fixed, k=5, dependent_var='y')
        avg_mseL = np.mean(test_errors)
        if regstr == 'Ridge':
            RidgeTestMSE_alpha.append(int(round(avg_mseL,0)))
        else:
            LassoTestMSE_alpha.append(int(round(avg_mseL,0)))
        #all_test_errors[alpha] = test_errors 
        if avg_mseL < reg[f'best_test_mse_{regstr}']:
            reg[f'best_alpha_{regstr}'] = alpha
            reg[f'best_test_mse_{regstr}'] = int(round(avg_mseL,0))
            reg[f'best_test_errors_{regstr}'] = test_errors
            reg[f'best_train_errors_{regstr}'] = train_errors
            reg['best_model_params'] =  model_params
            
    # for alpha in alphas_Ridge:
    #     alphaloop(alpha, best_parameters_Ridge, 'Ridge', ridge_wrapper)    
    func = partial(alphaloop, reg=best_parameters_Ridge, regstr='Ridge', wrapper=ridge_wrapper)
    apply = np.vectorize(func) #Is a for loop in disguise
    apply(alphas_Ridge ) 
    
    df=pd.DataFrame(RidgeTestMSE_alpha[1:],index=alphas_Ridge)
    df.plot( style='-o',label=['mse'])
    plt.title('AVG MSE by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.legend(['MSE'])
    plt.show()    
    
    
    plot_errors(best_parameters_Ridge['best_train_errors_Ridge'], best_parameters_Ridge['best_test_errors_Ridge'],5, 'Ridge')
             
    print(f"Best alpha Ridge: {best_parameters_Ridge['best_alpha_Ridge']}, MSE: {best_parameters_Ridge['best_test_mse_Ridge']}")
    #Loop checks if current iteration over alpha has a lower average value of test MSE over the ones preceding
    #It will substitute the value if true
#%% Lasso Cell
with timer("Lasso Count"): #starts time count
    alphas_Lasso = np.linspace(0.1,50,41)
    best_parameters_Lasso = {
       'best_test_mse_Lasso': float('inf'),
     }

    func = partial(alphaloop, reg=best_parameters_Lasso, regstr='Lasso', wrapper=lasso_wrapper)
    apply= np.vectorize(func)
    apply(alphas_Lasso)       

    df2=pd.DataFrame(LassoTestMSE_alpha[1:],index=alphas_Lasso)
    df2.plot( style='-o',label=['mse'])
    plt.title('AVG MSE by Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('MSE')
    plt.legend(['MSE'])
    plt.show()    

    plot_errors(best_parameters_Lasso['best_train_errors_Lasso'], best_parameters_Lasso['best_test_errors_Lasso'],5, 'Lasso')
         
    print(f"Best alpha Lasso: {best_parameters_Lasso['best_alpha_Lasso']}, MSE: {best_parameters_Lasso['best_test_mse_Lasso']}") 
#%%
with timer("simulation-loop"):
    datasets = [simulate_data(2400, 600, 600, 4, True_Parameters['true_betas'], True_Parameters['error_betas'], True_Parameters['sd_values'], True_Parameters['sd_values_rnd']) for _ in range(10)]
    best_alphas_ridge = []
    best_alphas_lasso = []
    best_mse_ridge = []
    best_mse_lasso = []
    
    # Loop over datasets
    for dataset in datasets:
        # Best alpha for Ridge
        best_parameters_Ridge = {'best_test_mse_Ridge': float('inf')}
        func_ridge = partial(alphaloop, reg=best_parameters_Ridge, regstr='Ridge', wrapper=ridge_wrapper)
        list(map(func_ridge, alphas_Ridge))
    
        # Best alpha for Lasso
        best_parameters_Lasso = {'best_test_mse_Lasso': float('inf')}
        func_lasso = partial(alphaloop, reg=best_parameters_Lasso, regstr='Lasso', wrapper=lasso_wrapper)
        list(map(func_lasso, alphas_Lasso))
    
        # Append results
        best_alphas_ridge.append(best_parameters_Ridge['best_alpha_Ridge'])
        best_alphas_lasso.append(best_parameters_Lasso['best_alpha_Lasso'])
        best_mse_ridge.append(best_parameters_Ridge['best_test_mse_Ridge'])
        best_mse_lasso.append(best_parameters_Lasso['best_test_mse_Lasso'])
    
        # Print progress
        print(f"Completed dataset {len(best_alphas_ridge)} out of {len(datasets)}")
    
    # Print results
    print("\nBest alphas for Ridge:")
    for alpha in best_alphas_ridge:
        print(alpha)
    
    print("\nBest alphas for Lasso:")
    for alpha in best_alphas_lasso:
        print(alpha)

        
    
    
    

