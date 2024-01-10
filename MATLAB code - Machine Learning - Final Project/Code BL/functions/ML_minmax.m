% ML_minmax - Absolute min&max of a matrix of whathever dimension (it accounts for NaN)

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function M=ML_minmax(x)
x(isinf(x))=NaN; 
J=length(size(x));
x1=x; x2=x;
for jj=1:J;  x1=nanmin(x1);   x2=nanmax(x2);   end
M=[x1 x2];
