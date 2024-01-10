import matplotlib.pyplot as plt

def plot_errors(train_errors, test_errors, k, reg_method):
    """
    Plot train vs test errors.
    
    Parameters:
    - train_errors: List (Vector) of training errors
    - test_errors: List (Vector) of test errors
    - k: Number of folds
    - reg_method: Regression method name, used in the title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, k+1), train_errors, '-o', label='Train MSE')
    plt.plot(range(1, k+1), test_errors, '-o', label='Test MSE')
    plt.xticks(range(1, k+1))  # Set x-axis ticks to be whole numbers
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # Format y-axis to three decimal points
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f'Train vs Test MSE for {k}-folds - {reg_method}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

