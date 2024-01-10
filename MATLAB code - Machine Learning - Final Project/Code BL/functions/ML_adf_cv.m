% ML_adf_cv - Critical values for the Augmented Dickey Fuller test
%           taken from the RATS procedure URADF.SRC by Norman Morin, 7/19/2004
%
% -----------------------------------------------------------------------------------------
% [cv]=ML_adf_cv(T,k,det)
%   cv = Critical Values
%   T = number of observation of the variable to be tested
%   k = number of differences included in the regression
%   det = deterministic part of the model (0 = no det; 1 = constant; 2 = constant & trend)
% -----------------------------------------------------------------------------------------

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function [cv]=ML_adf_cv(T,k,det)

nobs=T-k-2;

if det==0;
tcv=[ -2.66 -1.95 -1.60 
	  -2.62 -1.95 -1.61 
	  -2.60 -1.95 -1.61 
	  -2.58 -1.95 -1.62 
	  -2.58 -1.95 -1.62 
	  -2.58 -1.95 -1.62 ];
end;  

if det==1;
tcv=[ -3.75 -3.00 -2.63 
	  -3.58 -2.93 -2.60 
	  -3.51 -2.89 -2.58 
	  -3.46 -2.88 -2.57 
	  -3.44 -2.87 -2.57 
	  -3.43 -2.86 -2.57 ];
end;

if det==2;
tcv=[ -4.38 -3.60 -3.24 
	  -4.15 -3.50 -3.18 
	  -4.04 -3.45 -3.15 
	  -3.99 -3.43 -3.13 
	  -3.98 -3.42 -3.13 
	  -3.96 -3.41 -3.12 ];
end;  

if nobs<25;   cv=tcv(1,:); end;
if nobs<50;   cv=tcv(2,:); end;
if nobs<100;  cv=tcv(3,:); end;
if nobs<250;  cv=tcv(4,:); end;
if nobs<=500; cv=tcv(5,:); end;
if nobs>500;  cv=tcv(6,:); end;