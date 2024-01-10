from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def Lasso_a(data, alpha=1.0, dependent_var='y'):
    """
    Estimate a Lasso regression using all variables in the dataset except for the specified dependent variable.
    
    Parameters:
    - data: DataFrame containing the dataset
    - alpha: Regularization hyper-parameter for Lasso regression
    
    Returns:
    - Lasso regression results
    """
    
    y = data[dependent_var].values
    X = (data.drop(dependent_var, axis=1))  
    
    lasso = Lasso(alpha=alpha)  #Performs Lasso regression
    lasso.fit(X, y)
    
    predictions = lasso.predict(X)
    mse = mean_squared_error(y, predictions)  #Evaluate the mse
    
    # Extract feature names and associate with coefficients
    feature_names = ['Intercept'] + X.columns.tolist()
    coefficients = [lasso.intercept_] + list(lasso.coef_)

    params = dict(zip(feature_names, coefficients))
    
    return {
        'model': lasso,
        'params': params,
        'mse': mse
    }

