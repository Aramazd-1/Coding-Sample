from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm    
def k_fold_cv(data, regression_func, k=5, dependent_var='y'):
        """
        Perform k-fold cross-validation for a given regression function.
        
        Parameters:
        - data: DataFrame containing the dataset
        - regression_func: The regression function to use for estimation
        - k: Number of folds for cross-validation
        - dependent_var: Name of the dependent variable column in the data
        - **kwargs: allows for keyword argument to be passed (e.g from the additional regression_func)
        Returns:
        - train_errors: List of training errors for each fold
        - test_errors: List of test errors for each fold
        """
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        train_errors = []  # Training MSE for each fold
        test_errors = []   # Test MSE for each fold
        models_params = []
        model_summaries = []
        
        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            
            
            # Check if the regression function is the OLS function
            if regression_func.__name__ == 'ols':
                X_train = sm.add_constant(train_data.drop(dependent_var, axis=1))
                X_test = sm.add_constant(test_data.drop(dependent_var, axis=1))
            else:
                 X_train = train_data.drop(dependent_var, axis=1)
                 X_test = test_data.drop(dependent_var, axis=1)
            # Train the model using the regression function
            results = regression_func(train_data, dependent_var)
                
            train_predictions = results['model'].predict(X_train)
            train_mse = round(mean_squared_error(train_data[dependent_var], train_predictions),3)

            test_predictions = results['model'].predict(X_test)
            test_mse = round(mean_squared_error(test_data[dependent_var], test_predictions), 3)
            
            train_errors.append(train_mse)
            test_errors.append(test_mse)
            if hasattr(results['model'], 'params'):
             models_params.append(results['model'].params)
             model_summaries.append(results['model'].summary().as_text())
            else: 
                models_params.append(results['params'])
                model_summaries.append("Sklearn model; summary not available")  # Placeholder text

        
        min_test_error_index = test_errors.index(min(test_errors))
        best_model_params = models_params[min_test_error_index]
        best_model_summary = model_summaries[min_test_error_index]
            
         
    
        return train_errors, test_errors, best_model_params, best_model_summary
    
    
