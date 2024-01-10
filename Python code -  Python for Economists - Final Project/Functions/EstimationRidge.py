from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
def Ridge_a(data, alpha=1.0, dependent_var='y'):
    """
    Estimate a Ridge regression using all variables in the dataset except for the specified dependent variable.
    
    Parameters:
    - data: DataFrame containing the dataset
    - alpha: Regularization hyper-parameter for Ridge regression
    
    Returns:
    - Ridge regression results
    """
    
    y = data[dependent_var].values  
    X = (data.drop(dependent_var, axis=1))  
    
    ridge = Ridge(alpha=alpha) 
    ridge.fit(X, y)            
    
    predictions = ridge.predict(X) 
    mse = mean_squared_error(y, predictions) 
    
    
    # Extract feature names and associate with coefficients
    feature_names = ['Intercept'] + X.columns.tolist()
    coefficients = [ridge.intercept_] + list(ridge.coef_)

    params = dict(zip(feature_names, coefficients))
    return {
        'model': ridge,
        'params': params,
        'mse': mse
    }