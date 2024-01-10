%  ML_round - Rounds to specific decimals
% 
%   ML_round(x,ndec,jj)
%   ndec = decimal place at which to round
%   jj = {0=closest (default),1=down,2=up)
% 

% Matteo Luciani (matteoluciani@yahoo.it)

  function y=ML_round(x,ndec,jj)

if nargin==2; jj=0; end
zz=10^ndec;

if jj==0; y=round(x*zz)/zz;
elseif jj==1; y=floor(x*zz)/zz;
elseif jj==2; y=ceil(x*zz)/zz;
end
