import numpy as np
import pandas as pd

def simulate_rel (num_independent_vars=20, num_random_vars=5, df=5):
    """
    Parameters:
    - num_independent_vars: Number of independent variables
    - num_random_vars: Number of variables in dataset with no correlation to the DGP
    
    Returns:
    -True_Parameters: true_betas sd_values error_betas
    
    
    """
    degrees_of_freedom = df #We made a lot of test and keeping it small is a good idea
    sd_values = np.random.chisquare(degrees_of_freedom, num_independent_vars)
    true_betas = np.random.randn(num_independent_vars)
    error_betas = np.random.randn(num_independent_vars)
    sd_values_rnd = np.random.chisquare(degrees_of_freedom, num_random_vars)
    True_Parameters = {
        'true_betas': true_betas,
        'error_betas': error_betas,
        'sd_values': sd_values,
        'sd_values_rnd': sd_values_rnd
    }
    
    return True_Parameters
def simulate_data(n=1000,num_independent_vars=20, num_random_vars=5, noise_level=1,true_betas=0,error_betas=1,sd_values=1,sd_values_rnd=1):
    """
    Simulate a dataset given parameters

    Parameters:
    - n: Number of observations
    - num_independent_vars: Number of independent variables
    - num_random_vars: Number of variables in dataset with no correlation to the DGP
    - noise_level: Standard deviation of the noise
    -True_parameters from before
    Returns:
    - DataFrame with simulated data
    """
    # Simulate independent variables
    X = np.random.randn(n, num_independent_vars)
    # Random SD values for each independent variable
    for i in range(num_independent_vars):
        X[:, i] *= sd_values[i]
    # Compute standard deviation of the error term for each observation
    error_std_dev = noise_level + X @ error_betas
    error_std_dev = np.abs(error_std_dev)  # Ensure all values are positive
    # Actual Generation of heteroskedastic errors
    errors = np.array([np.random.normal(0, std) for std in error_std_dev])

    # True Linear relationship 
    y = 5 + X @ true_betas + errors

    # Add random variables that aren't related to y
    random_vars = np.random.randn(n, num_random_vars)
    for i in range(num_random_vars):
        random_vars[:, i] *= sd_values_rnd[i]

    # Combine all variables into a DataFrame
    data = pd.concat([
        pd.DataFrame(
            X, columns=[f'x{i+1}' for i in range(num_independent_vars)]),
        pd.DataFrame({'y': y}),
        pd.DataFrame(random_vars, columns=[
                     f'random_var{i+1}' for i in range(num_random_vars)])
    ], axis=1)

    return data
#