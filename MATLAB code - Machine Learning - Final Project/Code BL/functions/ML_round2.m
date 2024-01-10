% ML_round2 - Rounds to specific decimals
% 
% ML_round2(x,jj,ndec,nint)
%   jj = rounds up or down
% ndec = decimal place at which to round
% nint = rounds at closest integer which is a multiple of nint
% 

% Matteo Luciani (matteoluciani@yahoo.it)

function y=ML_round2(x,jj,ndec,nint)
dd=10^ndec;
nn=nint;
if jj==1; y=nn*floor(dd*ML_min(x)/nn)/dd;
elseif jj==2; y=nn*ceil(dd*ML_min(x,2)/nn)/dd;
end