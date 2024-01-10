% ML_DynamicFactorEM_GDO_TV - ML estimation of NSDFM with LT and TV deterministic components
%                         the linear trend is treated as state
% 
% [xitT,PtT,PtTm,xitt,xittm,Ptt,Pttm,A,Lambda,R,Q,b1,mu]=...
%       ML_DynamicFactorEM_GDO_TV(Y,F00,P0s,A,Lambda,R,Q,s,q,d,p,mu,type,maxiter,tresh,cc)
%
%       Y - data with deterministic component
%     F00 - initial values for the states
%     P0s - initial variance for the states
%       A - VAR(1) for the states
%  Lambda - Loadings
%       R - covariance matrix errors
%       Q - covariance matrix shocks
%       s - number of lags in the factor loadings
%       q - number of shocks
%       p - number of lags VAR factors
%      mu - constant in Factors VAR, and slope for LT
%    type - identify type of state (TV,RW,WNwLT,RWwLT)
% maxiter - max number iterations
%   tresh - treshold for stopping algorithm
%      cc - # of obs eliminated before estimateing parameters
%     GDO - if ==1 impose restrictions GDO:=GDP+GDI. GDP is variable ...
%                  ... number 1 while GDI is variable number 2
% 
% Reference: Barigozzi, M. and Luciani, M. (2019). 
%  "Quasi maximum likelihood estimation of nonstationary large approximate 
%    dynamic factor models". arXiv
% 

% Matteo Luciani (matteoluciani@yahoo.it)

function [xitT,PtT,PtTm,xitt,xittm,Ptt,Pttm,A,Lambda,R,Q,b1,mu]=...
    ML_DynamicFactorEM_GDO_TV(Y,F00,P0s,A,Lambda,R,Q,s,q,p,mu,type,maxiter,tresh,cc,GDO)

try isnan(GDO); catch, GDO=1; end

OPTS.disp = 0;
[T, N]=size(Y);                                                             % size of the panel
s1=s+1;
r=q*(s1);                                                                   % number of static factors
pq=q*p;
pqs=max(pq,r);

TT=(1:T)';                                                                  % time trend
isxi=find(type==1)+pqs;                                                     % identifies states that are idiosyncratic components
istrend=find(mu~=0); istrend(1:q)=[];                                       % identifies states with linear trend
isc2=isxi(~ismember(isxi,istrend));                                         % identifies states that are idio RWnLT
isc3=istrend(~ismember(istrend,isxi));                                     	% identifies states that are linear trends
isc4=isxi(ismember(isxi,istrend));                                          % identifies states that are idio RWwLT
J2=find(sum(Lambda(:,isc2),2)==1); nJ2=ismember(1:N,J2);                    % identifies variables that have idio RWnLT 
J3=find(sum(Lambda(:,isc3),2)==1); nJ3=ismember(1:N,J3);                    % identifies variables that have idio WNwLT 
J4=find(sum(Lambda(:,isc4),2)==1); nJ4=ismember(1:N,J4);                    % identifies variables that have idio RWwLT 
isTV=find(type==10)+pqs;                                                    % TV deterministic components

%%% =================== %%%
%%%  Start with E-Step  %%%
%%% =================== %%%
[xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter2(F00,P0s,Y,A,Lambda,R,Q,mu);   % Kalman Filter                           
[xitT,PtT,PtTm]=ML_KalmanSmoother2(A,xitt',xittm',Ptt,Pttm,Lambda,R);       % Kalman Smoother
F00=xitT(1,:)'; P00=PtT(:,:,1);                                             % initial conditions
F=xitT(:,1:q);                                                              % factors
FF=[]; for ss=0:s; FF=cat(2,FF,F(s-ss+1:end-ss,:)); end                     % lagged factors 0:s
LT=zeros(T,N); EE=zeros(T,N);                                               % preallocates
for ii=1:length(isTV) % --------------------------------------------------- % TV slopes or means
    id{ii}=find(Lambda(:,isTV(ii)));
    LT(:,id{ii})=xitT(:,isTV(ii))*ones(1,length(id{ii}));                     
end
LT(:,nJ3)=xitT(:,isc3);                                                     % linear trend for WNwLT
LT(:,nJ4)=TT*mu(isc4)';                                                     % linear trend for RWwLT
EE(:,nJ2)=xitT(:,isc2);                                                     % idio for RWnLT
EE(:,nJ4)=xitT(:,isc4)-LT(:,nJ4);                                           % idio for RWwLT
b1=zeros(N,1); b1([find(nJ3) find(nJ4)])=mu([isc3; isc4]);                  % slope linear trend
loglik1=loglik;                                                             % likelihood

disp('Running EM algorithm: ')

disp(['0: ' num2str(loglik)])
for jj=1:maxiter         
    
    %%%% ======================== %%%%
    %%%% ====                ==== %%%%
    %%%% ====    M - Step    ==== %%%%
    %%%% ====                ==== %%%%
    %%%% ======================== %%%%
    
    %%% =========================== %%%
    %%% 1: Estimate factor Loadings %%%
    %%% =========================== %%%
    yy=ML_center(Y)';                                                       % endogenous variables,     
    xx=cat(2,zeros(r,s),ML_center(FF)');                                    % exogenous variables, aka the factors    
    zz=ML_center(EE)';                                                      % I(1) idio need to be subtracted
    ww=ML_center(LT)';                                                      % Linear trend need to be subtracted
    num=zeros(N,r); den=zeros(r,r);                                         % preallocates
    for tt=cc+1:T  % ------------------------------------------------------ % \sum_{t=1}^T
        num=num+(yy(:,tt)-zz(:,tt)-ww(:,tt))*xx(:,tt)';                     % (y_t - \xi_t) F_{t|T}'
        den=den+xx(:,tt)*xx(:,tt)'+PtT(1:r,1:r,tt);                         % F_{t|T}F_{t|T}'+P_{t|T}
    end    
    L=num/den;                                                              % factor loadings       
    if GDO==1; L(1:2,:)=repmat(mean(L(1:2,:)),2,1); end                     % GDO loadings
    Lambda(1:N,1:r) = L;                                                    % store factor loadings

    %%% ============================== %%%
    %%% 2: Estimate constant and slope %%%
    %%% ============================== %%%
    yy=Y';                                                                  % endogenous variables, 
    xx=[ones(1,T); TT'];                                                    % exogenous variable, aka time trend and constant (for good measure)
    zz=EE';                                                                 % xi_t ~ I(1) need to be subtracted                
    ww=cat(2,zeros(N,s),L*FF');                                             % \chi_t need to be subtracted    
    JJ=[find(nJ3) find(nJ4)];                                               % identifies position of the trends
    num=zeros(sum(nJ3+nJ4),2); den=zeros(1,1);                              % preallocates
    for tt=cc+1:T  % ------------------------------------------------------ % \sum_{t=1}^T
        num=num+(yy(JJ,tt)-zz(JJ,tt)-ww(JJ,tt))*xx(:,tt)';                  % (y_t-chi_t-\xi_t) * [1 t]
        den=den+xx(:,tt)*xx(:,tt)';                                         % [1 t]'[1 t]
    end      
    temp=num/den;                                                           % constant and slope
    b0=b1;                                                                  % slopes in (jj-1)-th iteration
    b1([find(nJ3) find(nJ4)])=temp(:,2); mu([isc3;isc4])=temp(:,2);         % slopes in jj-th iteration    
    if GDO==1; b1(find(nJ3(1:2)))=mean(b1(find(nJ3(1:2)))); end             % slope for GDO        
    F00(isc3)=temp(1:sum(nJ3),2);                                           % starting value for WNwLT
    if GDO==1; F00(isc3(1:2))=mean(F00(isc3(1:2))); end                     % starting value GDO            
    F00(isc4)=F00(isc4)+b1(find(nJ4))-b0(find(nJ4));                        % starting value for RWwLT       
  
    
    %%% ================================== %%%
    %%% 3: Estimate VAR on common Factors  %%%
    %%% ================================== %%%
    yy=cat(2,zeros(q,1),F(2:T,:)');                                         % F_t
    xx=cat(1,ones(1,T),cat(2,ML_lag(xitT(:,1:pq),1)',zeros(pq,1)));         % F_{t-1}    
    EF1=zeros(q,pq+1); EF1F1=zeros(pq+1,pq+1); EF=zeros(q,q);               % initialize
    for tt=cc+2:T  % ------------------------------------------------------ % \sum_{t=1}^T
        EF1=EF1+yy(:,tt)*xx(:,tt-1)'+[zeros(q,1) PtTm(1:q,1:pq,tt)];        % E(F_t F_{t-1})
        EF1F1=EF1F1+xx(:,tt-1)*xx(:,tt-1)'+blkdiag(0,PtT(1:pq,1:pq,tt-1));  % E(F_{t-1} F_{t-1})
        EF=EF+yy(:,tt)*yy(:,tt)'+PtT(1:q,1:q,tt);                           % E(F_t F_t) 
    end           
    temp=EF1/EF1F1;
    A(1:q,1:pq)=temp(:,2:end);                                              % parameter VAR factors    
    mu(1:q)=temp(:,1);                                                      % constant VAR factors    
    Q(1:q,1:q) = (EF - temp*EF1') / (T-cc);                                 % covariance    

    %%% =================================================== %%%
    %%% 4: Estimate variance for RW Idiosyncratic Component %%%
    %%% =================================================== %%%
    JJ=[find(nJ2) find(nJ4)]; JJ2=[isc2; isc4];
    EE2=ML_center(EE(:,JJ));
    yy=cat(2,zeros(length(JJ),1),EE2(2:T,:)');                              % \xi_t
    xx=cat(2,ML_lag(EE2,1)',zeros(length(JJ),1));                           % \xi_{t-1}
    for nn=1:sum(nJ2+nJ4)  % ---------------------------------------------- % Only for I(1) idio
        Eee1=0; Ee1e1=0; Eee=0;                                             % initialize
        for tt=cc+2:T  % -------------------------------------------------- % \sum_{t=1}^T
            Eee1=Eee1+yy(nn,tt)*xx(nn,tt-1)'+PtTm(pqs+nn,pqs+nn,tt);        % E(\xi_t \xi_{t-1})
            Ee1e1=Ee1e1+xx(nn,tt-1)*xx(nn,tt-1)'+PtT(pqs+nn,pqs+nn,tt-1);   % E(\xi_{t-1} \xi_{t-1})
            Eee=Eee+yy(nn,tt)*yy(nn,tt)'+PtT(pqs+nn,pqs+nn,tt);             % E(\xi_t \xi_t)
        end
        Q(JJ2(nn),JJ2(nn))=(Eee + Ee1e1 - 2*Eee1') / (T-cc);                % variance of nn-th idiosyncratic shock
    end   
    
    %%% ==================================================== %%%
    %%% 5: Estimate variance for TV deterministic components %%%
    %%% ==================================================== %%%    
    nnTV=isTV+(type(isTV-pqs+1)==3);    
    EE2=ML_center(xitT(:,nnTV));
    yy=cat(2,zeros(length(isTV),1),EE2(2:T,:)');                            % \xi_t
    xx=cat(2,ML_lag(EE2,1)',zeros(length(isTV),1));                         % \xi_{t-1}  
    for nn=1:length(isTV); nTV=nnTV(nn);
        Eee1=0; Ee1e1=0; Eee=0;                                             % initialize
        for tt=cc+2:T  % -------------------------------------------------- % \sum_{t=1}^T
            Eee1=Eee1+yy(nn,tt)*xx(nn,tt-1)'+PtTm(nTV,nTV,tt);              % E(\xi_t \xi_{t-1})
            Ee1e1=Ee1e1+xx(nn,tt-1)*xx(nn,tt-1)'+PtT(nTV,nTV,tt-1);         % E(\xi_{t-1} \xi_{t-1})
            Eee=Eee+yy(nn,tt)*yy(nn,tt)'+PtT(nTV,nTV,tt);                   % E(\xi_t \xi_t)
        end
        Q(nTV,nTV)=(Eee + Ee1e1 - 2*Eee1') / (T-cc);                        % variance of TV deterministic components
    end            
    
    %%% ============================== %%%
    %%% 6: Covariance Prediction Error %%%
    %%% ============================== %%%
    yy=Y';
    xx=xitT';
    R=zeros(N,N);                                                           % preallocates
    for tt=cc+1:T  % ------------------------------------------------------ % \sum_{t=1}^T
        eta=yy(:,tt)-Lambda*xx(:,tt);                                       % prediction error
        R = R+ eta*eta'+ Lambda*PtT(:,:,tt)*Lambda';                        % E(eta_t eta_t')
    end
    R = diag(diag(R/(T-cc)));
    
    %%%% ======================== %%%%
    %%%% ====                ==== %%%%
    %%%% ====    E - Step    ==== %%%%
    %%%% ====                ==== %%%%
    %%%% ======================== %%%%            
    [xitt,xittm,Ptt,Pttm,loglik]=ML_KalmanFilter2(F00,P00,Y,A,Lambda,R,Q,mu); % Kalman Filter
    [xitT,PtT,PtTm]=ML_KalmanSmoother2(A,xitt',xittm',Ptt,Pttm,Lambda,R);   % Kalman Smoother           
    F=xitT(:,1:q);                                                          % jj-th step factors    
    FF=[]; for ss=0:s; FF=cat(2,FF,F(s-ss+1:end-ss,:)); end                 % jj-th step lagged factors 0:s    
    for ii=1:length(isTV) % ----------------------------------------------- % jj-th TV slopes or means        
        LT(:,id{ii})=xitT(:,isTV(ii))*ones(1,length(id{ii}));
    end
    LT(:,nJ3)=xitT(:,isc3);                                                 % jj-th step linear trend for WNwLT
    LT(:,nJ4)=TT*mu(isc4)';                                                 % jj-th step linear trend for RWwLT
    EE(:,nJ2)=xitT(:,isc2);                                                 % jj-th step idio for RWnLT
    EE(:,nJ4)=xitT(:,isc4)-LT(:,nJ4);                                       % jj-th step idio for RWwLT
    F00=xitT(1,:)'; P00=PtT(:,:,1);                                         % initial conditions    

    %%%% ================================= %%%%
    %%%% ====                         ==== %%%%
    %%%% ====    Check convergence    ==== %%%%
    %%%% ====                         ==== %%%%
    %%%% ================================= %%%%
    delta_loglik = loglik - loglik1;                                        % logL(t) - logL(t-1)
    avg_loglik = (abs(loglik) + abs(loglik1) + 10^(-3))/2;                  % avg = (|logL(t)| + |logL(t-1)|)/2
    disp([num2str(jj) ': ' num2str(loglik) '  ' ...
        num2str(delta_loglik / avg_loglik)])
    if (abs(delta_loglik) / avg_loglik) < tresh; break; end                 % convergence if |f(t) - f(t-1)| / avg < threshold
    loglik1=loglik;                                                         % store log-likelihood
end

if jj==maxiter 
    disp('not converged'); 
else
    disp(['converged after ' num2str(jj) ' iterations'])
end
