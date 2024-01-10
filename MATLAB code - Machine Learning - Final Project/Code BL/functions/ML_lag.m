% ML_lag - Given x(t) produces x(t-j0) .... x(t-k)
%
% xx=ML_lag(x,k,j0)
% by default j0 is 1
%
% eg x=[x1 x2]
% xx=ML_lag(x,2)=[x1_{t-1} x1_{t-2} x2_{t-1} x2_{t-2}]
%

% Matteo Luciani (matteoluciani@yahoo.it)

function xx=ML_lag(x,k,j0)
[T, N] = size(x);

if nargin<3; j0=1; end;    
n=1;

if N*(k+1-j0)==0; 
    xx=[];
elseif T==1;
    xx=x;
else
    xx=zeros(T-k,N*(k+1-j0));
    
    for i=1:N;
        for j=j0:k,
            xx(:,n)=x(k+1-j:T-j,i);
            n=n+1;
        end;
    end;
end
    
    
    