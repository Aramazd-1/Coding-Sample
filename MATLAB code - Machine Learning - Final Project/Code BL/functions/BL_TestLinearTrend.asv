% BL_TestLinearTrend - test if there is a linear trend
% 
% J=BL_TestLinearTrend(y)
% 
% J=variables that need to be detrended
%
% In order to choose whether or not to de-trend a variable we apply the following procedure: 
% 	let $m_i$ be the sample mean of $\Delta y_{it}$, 
% 	$\gamma_{i}(j)$ be the auto-covariance of order $j$ of $\Delta y_{it}$, 
% 	and $\bar{\gamma}_i=\sqrt{\frac{1}{T}\sum_{j=-J}^J \gamma_i(j)}$, 
% 	then if  $\frac{|m_i|}{\bar{\gamma}_i}\ge 1.96$ we estimate $a_i$ and $b_i$ 
% 	from an OLS regression of $y_{it}$ on a constant and a time trend, 
% 	whereas if $\frac{|m_i|}{\bar{\gamma}_i}< 1.96$ we set $\wh{a}_i=m_i$ and $\wh{b}_i=0$.
% 

% Matteo Luciani (matteoluciani@yahoo.it)

function J=BL_TestLinearTrend(y)
[T,N]=size(y);
my=mean(y);                                                                 % Sample mean
K=round(sqrt(T)); K1=round(T^.6); K2=round(T^.4);                           % number of lags for which we compute the autocorrelation
gamma=NaN(K1+1,N);                                                          % preallocates
for ii=1:N;    
    for jj=0:K1;                                                            % take some extra cov just in case 
        x1=y(jj+1:end,ii); 
        xx1=x1-mean(x1);
        x2=y(1:end-jj,ii);
        xx2=x2-mean(x2);
        gamma(jj+1,ii)=(xx1'*xx2)/(T-jj-1);                                 % autocovariance
        clear x1 x2
    end    
end;

gammaS=sum(gamma(1:K+1,:))+sum(gamma(2:K+1,:));                             % this sum has to be positive
I=find(gammaS<0);                                                           % In finite sample it might be that 
for ii=I; jj=0;                                                             % this does not happen 
    while gammaS(ii)<0; jj=jj+1;  if K+jj>K1; break; end                    % e.g. with highly negative correlated variables                                                      
        gammaS(ii)=sum(gamma(1:K+1+jj,ii))+sum(gamma(2:K+1+jj,ii));         % to solve this, I extend K untill
    end                                                                     % the sum becomes positive
    if gammaS(ii)<0;                                                        % in case this does not work
        while gammaS(ii)<0; jj=jj-1;  if K+jj<K2; break; end                % I reduce the number of cov
            gammaS(ii)=sum(gamma(1:K+1+jj,ii))+sum(gamma(2:K+1+jj,ii));     %
        end                                                                 %
    end
    if gammaS(ii)<0; disp(['Check variable ' num2str(ii)]); end             % If the problem persists, then there is some issue
end

gammabar= sqrt(gammaS/T);                                                   % standard deviation sample mean
J=(abs(my)./gammabar)>=1.96;


