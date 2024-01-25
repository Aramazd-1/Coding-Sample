import numpy as np
from numpy.linalg import matrix_power, inv
from Structural_Functions import select_blocks, Companion_Matrix_compute
from VAR_estimation import estimate_var_model_const_trend_ols_boot
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
def MBB_parallel(T,p,M,blocks,block_length,Sample_boot,Non_ar_params,AR_matrices,hor):
    Residuals_boot = select_blocks(blocks, int(T / block_length) + 1)[:T, :]  # Randomly samples block with replacement
    Residuals_boot = Residuals_boot - np.mean(Residuals_boot, axis=0, keepdims=True)  # We center residuals
    instrument_boot = Residuals_boot[:, M]  # Selects the last column to be the Residuals of the instrument
    Residuals_boot = Residuals_boot[:, :M]

    for i in range(p,T): Sample_boot[i] = Non_ar_params[:,0].T + (Non_ar_params[:,1].T)*i + sum(AR_matrices[j] @ Sample_boot[i-j-1].T for j in range(0, p)) + Residuals_boot[i-2,:]           
 

    # Estimation of the model and IRF
    Sigma_u_boot, AR_matrices_boot, Residuals_boot = estimate_var_model_const_trend_ols_boot(Sample_boot, p=p, noprint=True)
    phi_hat_boot = 1 / (T - p) * np.dot(Residuals_boot.T,instrument_boot)  # We compute the sample covariance between the instrument and the Residual
    rho_squared_hat_boot = phi_hat_boot.T @ np.linalg.inv(Sigma_u_boot) @ phi_hat_boot
    k_hat_boot = rho_squared_hat_boot ** (-1 / 2) * phi_hat_boot
    if k_hat_boot[0] < 0 : k_hat_boot=k_hat_boot*(-1)
    Companion_Matrix_boot = Companion_Matrix_compute(AR_matrices_boot, M)

    IRF_hat_matrix_boot = np.zeros((hor + 1, M))
    if p>1 :R_selection= np.hstack([np.eye(M), np.zeros((M, M*(p-1)))])
    else: R_selection = np.eye(M)
    for h in range(0, hor + 1):IRF_hat_matrix_boot[h, :] = (R_selection @ matrix_power(Companion_Matrix_boot, h) @ R_selection.T @ k_hat_boot).T
    return IRF_hat_matrix_boot

def exec_MBB_parallel(T, p, M, blocks, block_length, Sample_boot, Non_ar_params, AR_matrices, hor, nboot):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(MBB_parallel, T, p, M, blocks, block_length, Sample_boot, Non_ar_params, AR_matrices, hor) for _ in range(nboot)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    return results