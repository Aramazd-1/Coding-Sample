% ML_Standardize - Standardize Variables
%
% [y M s] = ML_Standardize(x)
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function [y, M, s] = ML_Standardize(x)

T=size(x,1);
s = nanstd(x);
M = nanmean(x);
ss = ones(T,1)*s;
MM = ones(T,1)*M;
y = (x-MM)./ss;