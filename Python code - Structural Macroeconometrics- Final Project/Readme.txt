The code depends on MatLab.api (matlab.engine), more at https://it.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
MATLAB dependencies: Econometrics toolbox, Statistics and Machine Learning toolbox 
PYTHON dependencies: pandas, matplotlib, numpy, statsmodels, time,  matlab.engine, scipy, concurrent.futures

Note: 
    - MATLAB DEPENDENCIES can be skipped by simply removing the import "engine.matlab" and not running the related cell. 
	- One can also avoid using parallel computing dependencies by not running the parallel computing cell and not importing concurrent.futures