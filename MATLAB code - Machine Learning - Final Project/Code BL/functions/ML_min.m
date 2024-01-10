% ML_min - Absolute min/max of a matrix of whathever dimension (it accounts for NaN)
% 
% x = ML_min(x,type)
%   type = 1 takes min (default)
%   type = 2 takes max
% 

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function x=ML_min(x,type)
x(isinf(x))=NaN;
if nargin==1; type=1; end 

J=max(size(size(x)));                      % number of dimensions of matrix x

if type==1; for jj=1:J; x=nanmin(x); end   % take the minimum
else        for jj=1:J; x=nanmax(x); end   % take the maximum
end


