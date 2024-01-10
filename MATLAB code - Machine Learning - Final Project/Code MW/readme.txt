The zip folder contains the replication files of Morley, Rodriguez Palenzuela, Sun & Wong (2023), “Estimating the Euro Area Output Gap Using Multivariate Information and Addressing the COVID- 19 Pandemic”.

The MATLAB code will reproduce Figures 1-8 from the paper. The main code executes from running the replication_EER.m file. If you want to replicate a Figure in the paper, turn options.Figure to 1 (e.g. options.Figure1 = 1). Otherwise, turn this to zero (or any other number).

Note that some of these Figures require multiple models to be estimated, so can take quite some time.
You can reduce the time by reducing the number of MCMC draws. Nonetheless, while 3000 or so draws from the posterior can get you results that look largely like the paper, many draws are required in order to nail the estimated output gap during COVID-19, especially given the different sequence of random numbers drawn, inaccuracy with the numerical approximations given a small number of draws etc. We find 15,000 draws, with 7,000 burn-in (as we do for all the results in the paper) produces results that are numerically more stable across multiple runs of the MCMC chain.

You should expect to take somewhere between 8-12 hours to estimate everything.

We thank Michele Lenza and Giorgio Primiceri for providing code to implement the COVID correction which you may recognize us adapting some of this code.

We cannot be held responsible for any losses which may arise from errors in the code.
If you find errors, please get in touch with us.
If you use the code, please cite the paper.

James Morley
Diego Rodriguez Palenzuela
Yiqiao Sun
Benjamin Wong

Jan 2023