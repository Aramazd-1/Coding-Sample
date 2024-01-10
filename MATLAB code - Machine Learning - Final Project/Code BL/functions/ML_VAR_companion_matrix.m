% ML_VAR_companion_matrix - build the companion matrix for a VAR
% PHI=ML_VAR_companion_matrix(AL,det);
% Y(t)=AL(:,:,1)*Y(t-1)+...+AL(:,:,p)*Y(t-p)+u(t)
% For a VAR(3) with k variables:
% | AL(:,:,1)   AL(:,:,2)   AL(:,:,3)   |
% | eye(k,k)    zeros(k,k)  zeros(k,k) |
% | zeros(k,k)  eye(k,k)    zeros(k,k) |
%

% Written by Matteo Luciani (matteoluciani@yahoo.it)

function PHI=ML_VAR_companion_matrix(A)

s=size(A);
if length(s)==1; PHI=A;                             % AR(1)
elseif length(s)==2;                        
    if s(2)==1; p=s(1,1);                           % Autoregressive model
        if  p==2; PHI=[A'; 1 0];                    % AR(2)   
        else PHI=diag(ones(1,p-1),-1); PHI(1,:)=A'; % AR(p)
        end
    else PHI=A; end                                 % VAR(1)
else    
    k=s(1,1); p=s(1,3);
    PHI=[];
    for i=1:p; PHI=[PHI A(:,:,i)]; end;
    PHI=[PHI;eye(k*(p-1),k*(p-1)) zeros(k*(p-1),k)];
end