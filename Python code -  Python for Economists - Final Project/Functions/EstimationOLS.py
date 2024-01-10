import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
def ols(data, dependent_var= 'y'):
    """
    Estimate an OLS regression using all variables in the dataset except for 'y'.
    
    Parameters:
    - data: DataFrame containing the dataset
    
    Returns:
    - OLS regression results
    # Esempio:
    estimation_result = ols(data)
 
    """
    
    y = data[dependent_var] # Dep variable
    X = data.drop(dependent_var, axis=1) # Ind variables (all columns except 'y')
    X = sm.add_constant(X) # Add a constant
    #Actual Estimation
    model = sm.OLS(y, X).fit()
    robust_results = model.get_robustcov_results(cov_type='HC3')
    
    return {
        'OLSparams': model.params,
        'robust_summary': str(robust_results.summary()),
        'fitted_values': model.fittedvalues,
        'model': model
    }
def predict_ols(model, data, dependent_var='y'):
    """
    Use estimated model to make predictions on new/test data.
    
    Parameters:
    - model: The fitted model object from statsmodels
    - data: DataFrame for prediction
    
    Returns:
    - Predicted values
    """
    
    X = sm.add_constant(data.drop(dependent_var, axis=1))  # Add a constant to the new data
    predictions = model.predict(X)
    mse = mean_squared_error(data[dependent_var], predictions)
    
    return {
        'fitted values': predictions,
        'mse': mse,
    }
