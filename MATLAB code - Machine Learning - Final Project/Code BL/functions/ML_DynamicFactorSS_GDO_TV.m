% ML_DynamicFactorSS_GDO_TV - State Space representation of NSDFM, dynamic loadings
%                           constraints for GDO
%                           linear trend as a state
%                           TV slope for GDO
%                           TV mean for UNRATE

% Matteo Luciani (matteoluciani@yahoo.it)

function [A,Q,Lambda,R,F00,P00,mu,type2]=ML_DynamicFactorSS_GDO_TV(AL,L,v,q,s,xi,F,diffuse,I0,I1,beta,A0,TV,star)

[T,N]=size(xi); [~,~,p]=size(AL); 
s1=s+1;
r=q*(s1);                                                                   % number of "static" factors
ps=max(p,s1);
pq=q*p;
pqs=max(pq,r);
A=zeros(pqs,pqs); I = eye(pqs,pqs); A(q+1:pqs,1:pqs)=I(1:pqs-q,1:end);      % Rewrite the model as a VAR(1)
A(1:pq,1:pq)=ML_VAR_companion_matrix(AL);                                   % ---------------------------------
Lambda = [L zeros(N,q*(ps-1))];                                             % Factor loadings  
Q = zeros(pqs,pqs); Q(1:q,1:q)=cov(v);                                      % variance of the common shocks
nu0=.99;                                                                    % this is a simplification: by default max root for I(1) states

%%% ================================ %%%
%%%  Variance of the common factors  %%%
%%% ================================ %%%
if nargin==5; diffuse=zeros(r,1); end

if sum(diffuse)==0
    P00 = reshape(inv(eye((pr)^2)-kron(A,A))*Q(:),pqs,pqs);                 % initial state covariance
else % -------------------------------------------------------------------- % This is a simplified way to put a diffuse prior
    nu=max(real(eig(A)));                                                   % max root of the VAR
    for pp=1:p; AL2(:,:,pp)=AL(:,:,pp)*(nu0/nu)^pp; end                     % Correct coefficients so that max root is nu0
    A2=A; A2(1:pq,1:pq)=ML_VAR_companion_matrix(AL2);                       % VAR in companion form    
    P00 = reshape(inv(eye(pqs^2)-kron(A2,A2))*Q(:),pqs,pqs);                % initial state covariance ==> vec(V) = (I- A \kron A) vec(Q), see Hamilton p.265
end
                         
F00=[]; for pp=1:ps; F00=cat(1,F00,F(ps+1-pp,:)'); end                      % Initial condition for the factor
mu=0*F00; mu(1:q)=A0(1,:)';                                                 % constant for the factors

%%% ========================== %%%
%%%  Idiosyncratic Components  %%%
%%% ========================== %%%

%%% Step 1: Determine which idio is I(1) and which I(0) %%%
idio=zeros(N,1);                                                            % if idio(ii)==1 ==> xi(ii) is I(1)
for nn = 1:N
    adf=ML_adf(xi(:,nn),-1,1);  cv = ML_adf_cv(T,1,0);                      % test for unit roots as in Bai & Ng (2004, ECTA)  
    if adf>cv(3); idio(nn)=1; end                                           % variables with an I(1) idio             
end
idio(I1)=1;  idio(I0)=0;                                                    % Overrule of the test
trend=ones(N,1); trend(beta(:,2)==0)=0;


for ii=1:N
    if idio(ii)==0      &&  trend(ii)==0;	type(ii)=0;                     % idio is WN and no trend
    elseif idio(ii)==1  &&  trend(ii)==0;	type(ii)=1;                     % idio is RW and no trend        
    elseif idio(ii)==0  &&  trend(ii)==1;   type(ii)=2;                     % idio is WN and Linear trend 
    elseif idio(ii)==1  &&  trend(ii)==1;   type(ii)=3;                     % idio is RW and Linear trend 
    end
end

R=star*eye(N);                                                              % Initial variance idiosyncratic component
type2=[];
e=ML_diff(xi);                                                              % \Delta \xi_t, for \xi~I(1) is idiosyncratic shocks
hasTV=cell2mat(TV.id);

for ii=1:N
    if ismember(ii,hasTV) % ----------------------------------------------- % it has TV deterministic component       
        for kk = 1:numel(TV.id)
            if ismember(ii,TV.id{kk})                
                jj=find(TV.id{kk}==ii);
                break
            end
        end                
        if strcmp(TV.Type{kk}(jj,:),'trend')
            [A1,Q1,C1,p00,f00,mu0,type0]=TV_Trend(TV.id{kk},beta,N,nu0,TV.q0(kk));
        elseif strcmp(TV.Type{kk}(jj,:),'mean')
            [A1,Q1,C1,p00,f00,mu0,type0]=TV_Mean(TV.id{kk},N,nu0,TV.q0(kk));            
        elseif strcmp(TV.Type{kk}(jj,:),'none ')
            A1=[]; C1=[]; Q1=[]; p00=[]; f00=[]; type0=[]; mu0=[];            
        end
        
        R(ii,ii)=var(xi(:,ii));                                                 % variance of the idio
    else % -------------------------------------------------------------------- % No TV trend
        if type(ii)==0 % ------------------------------------------------------ % idio is WN and no trend, no need for a state variable
            A1=[]; C1=[]; Q1=[]; p00=[]; f00=[]; type0=[]; mu0=[];
            R(ii,ii)=var(xi(:,ii));
        elseif type(ii)==1; % ------------------------------------------------- % idio is RW and no trend
            type0=1;
            mu0=0;
            A1=1;                                                               % Autoregressive states
            C1=zeros(N,1); C1(ii)=1;                                            % loadings
            Q1=var(e(:,ii));                                                    % variance of the shock
            p00=Q1/(1-nu0^2);                                                   % initial variance state
            f00=xi(1,ii);                                                       % initial value state
        elseif type(ii)==2; % ------------------------------------------------- % WN + linear trend, need two states for trend
            type0=2;
            mu0=beta(ii,2);
            A1=1;                                                               % Autoregressive states
            C1=zeros(N,1); C1(ii,1)=1;                                          % loadings
            Q1=0;                                                               % variance of the shocks
            p00=0;                                                              % initial variance state
            f00=beta(ii,1)+beta(ii,2);                                          % initial value state
            R(ii,ii)=var(xi(:,ii));                                             % variance of the idio
        elseif type(ii)==3; % ------------------------------------------------- % RW + linear trend, need two states for trend
            type0=1;
            mu0=beta(ii,2);
            A1=1;                                                               % Autoregressive states
            C1=zeros(N,1); C1(ii,1)=1;                                          % loadings
            Q1=var(e(:,ii));                                                    % variance of the shocks
            p00=Q1(1)/(1-nu0^2);                                                % initial variance state
            f00=e(1,ii)+beta(ii,1)+beta(ii,2);                                  % initial value state
        end
    end
    A=blkdiag(A,A1);
    Q=blkdiag(Q,Q1);
    Lambda=cat(2,Lambda,C1);
    P00=blkdiag(P00,diag(p00));
    F00 = cat(1,F00, f00');
    mu = cat(1,mu, mu0);      
    type2 = cat(1,type2, type0');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A1,Q1,C1,p00,f00,mu0,type0]=TV_Mean(id,N,nu0,q0) % -------------- % Time Varying mean for Unemployment rate
type0=10;                                                                   %
A1=1;                                                                       % Autoregressive states
C1=zeros(N,1); C1(id,1)=1;                                                  % loadings
Q1=q0;                                                                      % variance of the shocks
p00=Q1/(1-nu0^2);                                                           % initial variance state
mu0=0;                                                                      %
f00=0;                                                                      % initial value state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A1,Q1,C1,p00,f00,mu0,type0]=TV_Trend(id,beta,N,nu0,q0) % -------- % Time Varying Trend
type0=[10 3];                                                               %
A1=[1 1; 0 1];                                                              % Autoregressive states
C1=zeros(N,2); C1(id,1)=1;                                                  % loadings
Q1=blkdiag(0,q0);                                                           % variance of the shocks
temp=Q1(2,2)/(1-nu0^2); p00=[(temp+Q1(1,1))/(1-nu0^2) temp];                % initial variance state
mu0=[0; 0];                                                                 %
f00=[mean(beta(id,1)+beta(id,2))  mean(beta(id,2))];                        % initial value state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
