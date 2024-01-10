% ML_center - Demean variables

% Matteo Luciani (matteoluciani@yahoo.it)

function XC = ML_center(X)
T = size(X,1);
XC = X - ones(T,1)*(sum(X)/T); 