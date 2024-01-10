import numpy as np
import matlab.engine
import pandas as pd

def estimate_var_model(data,p):
    """
    

    Parameters
    ----------
    data :  Numpy df
    p : the number of lags

    Returns
    -------
    Const :  Constant for each equation
    Trend :  time trend for each equation
    logLikVAR : log likelihood associated to estimate VAR model
    Sigma_u : This is the variance-covariance matrix of residuals

    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    # Check the type of 'data' and convert to MATLAB compatible format
    if isinstance(data, pd.DataFrame):
        # Convert pandas DataFrame to MATLAB compatible format (list of lists)
        matlab_data = matlab.double(data.values.tolist())
        variable_names = data.columns.tolist()
        print("dataframe loaded")
    elif isinstance(data, np.ndarray):
        # Convert NumPy array to MATLAB compatible format
        matlab_data = matlab.double(data.tolist())
        variable_names = [f'Var{i+1}' for i in range(data.shape[1])]
        print("array loaded")
    else:
        raise ValueError("Input data must be a pandas DataFrame or a NumPy array.")
    
    
    # Set up VAR model parameters
    p = p  # Number of lags
    T = len(data) - p
    M = data.shape[1]
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
    Trend = eng.getfield(EstVAR,'Trend')
    ar_field = eng.getfield(EstVAR, 'AR')
    ar_matrices_dict = {}
    for i in range(p):
        ar_matrix_np = np.array(ar_field[i])
        ar_matrices_dict[f'AR{i+1}'] = ar_matrix_np
    
    # Covariance matrix of residuals
    residuals_array = np.array(Residuals._data).reshape(Residuals.size[::-1]).T
    Sigma_u = np.dot(residuals_array.T, residuals_array) / T
    Sigma_u_df = pd.DataFrame(Sigma_u, index=variable_names, columns=variable_names)
    # Stop MATLAB engine
    eng.quit()

    return  Const, Trend, logLikVAR, Sigma_u_df, ar_matrices_dict


