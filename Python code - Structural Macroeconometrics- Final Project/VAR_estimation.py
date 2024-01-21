import numpy as np
import matlab.engine
import pandas as pd
from statsmodels.tsa.api import VAR

def estimate_var_model_const_trend(data,eng, p,noprint= False):
    """
    NOTE: It assumes a matlab session is started, to do so type eng = matlab.engine.start_matlab()
    to quit the session: eng.quit() 
    Parameters
    ----------
    data :  Numpy df
    p : the number of lags
    noprint: if true prevents printing (useful when doing boot)
    Returns
    -------
    Non_ar_params: A pandas dataframe containing both the constant and the trend of each equation
    Sigma_u : This is the variance-covariance matrix of residuals
    ar_matrices_dict: A dictionary containing as many numpy arrays as the lags of the estimate model

    """
    #eng = matlab.engine.start_matlab()
    # Check the type of 'data' and convert to MATLAB compatible format
    
    if isinstance(data, pd.DataFrame):
            # Convert pandas DataFrame to MATLAB compatible format (list of lists)
            matlab_data = matlab.double(data.values.tolist())
            variable_names = data.columns.tolist()
            if noprint is False:
                print("dataframe loaded")
    elif isinstance(data, np.ndarray):
            # Convert NumPy array to MATLAB compatible format
            matlab_data = matlab.double(data.tolist())
            variable_names = [f'Var{i+1}' for i in range(data.shape[1])]
            if noprint is False:
                print("array loaded")
    else:
            raise ValueError("Input data must be a pandas DataFrame or a NumPy array.")
   
    
    # Set up VAR model parameters
    p = p  # Number of lags
    T = len(data) - p
    M = data.shape[1]
    k = M*2+ M*M*p
    if noprint is False:
        print(f'Number of variables:{M}')
        print(f'Effective sample {T}')
    # Initialize constant, trend, and AR matrices
    VAR_Const = eng.nan(M, 1)
    VAR_Trend = eng.nan(M, 1)
    VAR_Pi = eng.cell(1, p)
    for i in range(p):
        VAR_Pi[i] = eng.nan(M, M)

    # Define the VAR model
    VAR = eng.varm('Constant', VAR_Const, 'AR', VAR_Pi, 'Trend', VAR_Trend)
    
    # Estimate the VAR model
    EstVAR, EstSE, logLikVAR, Residuals = eng.estimate(VAR, matlab_data, nargout=4)
    
    # Extracting estimates
    Const = eng.getfield(EstVAR,'Constant')
    Const = np.array(Const).flatten() #We need to do this so we have a 1D numpy array and can put it into dataframe
    Trend = eng.getfield(EstVAR,'Trend')
    Trend = np.array(Trend).flatten()
    Non_ar_params = pd.DataFrame({'Const':Const,'Trend':Trend}) #A dataframe containing both the trend and the constant
    Non_ar_params.index = variable_names 
    
    ar_field = eng.getfield(EstVAR, 'AR')
    ar_matrices_dict = {}
    for _ in range(p):
        ar_matrix_np = np.array(ar_field[_])
        ar_matrices_dict[f'AR{_+1}'] = ar_matrix_np
        
    
    # Covariance matrix of residuals
    residuals_array = np.array(Residuals._data).reshape(Residuals.size[::-1]).T
    Sigma_u = np.dot(residuals_array.T, residuals_array) /(T-p-1) 
    Sigma_u_df = pd.DataFrame(Sigma_u, index=variable_names, columns=variable_names)
    
    # eng.quit() # Stop MATLAB engine after all computations are done
    return  Non_ar_params, Sigma_u_df, ar_matrices_dict, residuals_array


def estimate_var_model_const_trend_ols_boot(data, p, freq='MS',noprint=False):
    """
    Estimate a VAR model with a constant and a trend in the bootstrap case

    Parameters
    ----------
    orig_data: the original non boot data
    data : pandas DataFrame or numpy ndarray
        Time-series data.
    p : int
        Number of lags.
    noprint : bool, optional
        If True, suppresses printing. Useful for bootstrapping.

    Returns
    -------
    non_ar_params : pandas DataFrame
        DataFrame containing the constant and the trend of each equation.
    sigma_u : pandas DataFrame
        Covariance matrix of residuals.
    ar_matrices_dict : dict
        Dictionary containing AR matrices for each lag.
    """
   
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("Input data must be a pandas DataFrame or a numpy ndarray.")
        
    # Convert numpy array to pandas DataFrame if necessary
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    
    M = data.shape[1]
    model = VAR(data)
    results = model.fit(p, trend='ct')

    if not noprint:
        print(f'Number of variables: {data.shape[1]}')
        print(f'Effective sample: {len(data) - p}')
        print(results.summary())

    
    results.params = results.params[results.params.shape[0]-M*p:]
    ar_matrices_dict = {}
    ar_matrices_dict = {f'AR{i+1}': (results.params).T[:, i*M:(i+1)*M] for i in range(p)}
    
    sigma_u = results.sigma_u # Covariance matrix of residuals
    residuals_array = results.resid

    return sigma_u, ar_matrices_dict, residuals_array


# Example usage
def VAR_lag_selection(data, eng, max_p, noprint= False):
    """
    auxiliary function that estimates a var for a given lag order p on given data using const and trend.
    Returns the results
    """
    
    #eng = matlab.engine.start_matlab()
    # Check the type of 'data' and convert to MATLAB compatible format
    
    if isinstance(data, pd.DataFrame):
            # Convert pandas DataFrame to MATLAB compatible format (list of lists)
            matlab_data = matlab.double(data.values.tolist())
            print("dataframe loaded")
    elif isinstance(data, np.ndarray):
            # Convert NumPy array to MATLAB compatible format
            matlab_data = matlab.double(data.tolist())
            print("array loaded")
    else:
            raise ValueError("Input data must be a pandas DataFrame or a NumPy array.")
    
    criteria_dict = {'Lag': [], 'AIC': [], 'BIC': [],'HQC': [], 'LogLikelihood':[]}
    # Set up VAR model parameters
    for p in range(1, max_p+1):
        T = len(data) - p
        M = data.shape[1] if isinstance(data, pd.DataFrame) else data.shape[0]
        VAR_Const = eng.nan(M, 1)
        VAR_Trend = eng.nan(M, 1)
        VAR_Pi = eng.cell(1, p)
        k = M*2+ M*M*p
        for i in range(p):
            VAR_Pi[i] = eng.nan(M, M)
            
        VAR = eng.varm('Constant', VAR_Const, 'AR', VAR_Pi, 'Trend', VAR_Trend)
        EstVAR = eng.estimate(VAR, matlab_data, nargout=1)
        
        Results = eng.summarize(EstVAR)
        aic = eng.getfield(Results, 'AIC') +  (2*k*k+2*k)/(T-k-1) #Last term is correction for small sample
        bic = eng.getfield(Results, 'BIC') 
        lik = eng.getfield(Results, 'LogLikelihood')   
        hqc = -2*lik +2*k*np.log(np.log(T))
        
        criteria_dict['Lag'].append(p)
        criteria_dict['AIC'].append(aic)
        criteria_dict['BIC'].append(bic)
        criteria_dict['HQC'].append(hqc)
        criteria_dict['LogLikelihood'].append(lik)
        
    # Convert results dictionary to Pandas DataFrame
    criteria_df = pd.DataFrame(criteria_dict)
    criteria_df = criteria_df.set_index('Lag')
    #eng.quit() # Stop MATLAB engine after all computations are done
    return criteria_df
        
        
    