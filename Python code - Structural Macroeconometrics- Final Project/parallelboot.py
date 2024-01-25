with timer('Boot cellparallel'):
    block_length = 12
    Residuals_expanded = np.hstack((Residuals, instrument[p:]))  # we stack all residuals together to jointly boot them
    blocks = generate_blocks(Residuals_expanded, block_length=block_length)
    nboot = 100 # Number of boot iterations
    IRF_matrices_boot = np.zeros((nboot, hor + 1, M))  # 3D array to store Bootstrap results
    Sample_boot = np.zeros((T + p, M))  # We Initialize the sample
    Sample_boot[:p, :] = Var_data.iloc[:p].to_numpy()  # makes first p values equal to the original sample
    
    results = exec_MBB_parallel(T, p, M, blocks, block_length, Sample_boot, Non_ar_params, AR_matrices, hor, nboot)
    IRF_matrices_boot = np.stack (results, axis=0)