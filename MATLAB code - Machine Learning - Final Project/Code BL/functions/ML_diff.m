% ML_diff - First difference of the data;
% xx=ML_diff(x,jj,kk)
%
% by default takes first difference of the series;
% jj is optional if actived the result is annual difference:
%   jj=1 for quarterly data
%   jj=2 for monthly data
%

% Written by Matteo Luciani
% matteoluciani@.yahoo.it

function xx=ML_diff(x,jj,kk)

if isempty(x); xx=[]; return; end
[T, N, B] = size(x);
if nargin ==1; kk=1;
elseif nargin==2; if jj==1; kk=4; elseif  jj==2; kk=12; end;
end;
xx=NaN(T-kk,N,B);
for ii=1:N; xx(:,ii,:)=x(kk+1:T,ii,:)-x(1:T-kk,ii,:); end;
    