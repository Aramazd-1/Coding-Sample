function [logLik]=Likelihood_SVAR_AB(teta,Sigma_u)


% global log_lk_reduced
global T
global ParamNumberA
global ParamNumberB

A = [1, 0, -2.08; 
     0, 1, 0 
     NaN, NaN, 1];
% Define the B matrix
B = zeros(size(Sigma_u,1),size(Sigma_u,1));
    for c_par = 1 : size(ParamNumberB,1)
    B(ParamNumberB(c_par,1))=teta(c_par);   
    end
    
    for c_par = 1 : size(ParamNumberA,1)
    A(ParamNumberA(c_par,1))=teta(c_par+size(ParamNumberB,1));   
    end
    
    M=size(B,1);


    logLik = -0.5*T*M*(log(2*pi)) + 0.5*T*log(det( A\B * B'/A' )) + 0.5*T*trace( (A'/B') * (B\A) *Sigma_u); %the sign has been changed because the procedure minimizes 
             
end
