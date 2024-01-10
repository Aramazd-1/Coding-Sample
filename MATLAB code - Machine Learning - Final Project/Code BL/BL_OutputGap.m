clear all; close all; clc; 


addpath([pwd '\functions']); 
%% Make dataset
rawdata.nonFRED = xlsread('dataset.xlsx','non_FRED','B3:AA87');
%rawdata.nonFRED = xlsread('dataset.xlsx','non_FRED','B3:O87');
%load the transformation for the non FRED data
rawdata.tcode = xlsread('dataset.xlsx','non_FRED','B2:AA2');

for jj = 1:size(rawdata.nonFRED,2)
    if rawdata.tcode(jj) == 0 %level
        y(:,jj) = rawdata.nonFRED(2:end,jj); %This leads us to drop the first value due to missing values
    end

    if rawdata.tcode(jj) == 1 % log level
        y(:,jj) = 100*log(rawdata.nonFRED(2:end,jj));
    end

    if rawdata.tcode(jj) == 2 % log difference
        y(:,jj) = 100*diff(log(rawdata.nonFRED(:,jj)));
    end

    if rawdata.tcode(jj) == 3 % difference
        y(:,jj) = diff(rawdata.nonFRED(:,jj));
    end
end

%load FRED data
rawdata.FRED = xlsread('dataset.xlsx','FRED','B7:F99');
[100*diff(log(rawdata.FRED(:,1))) 100*log(rawdata.FRED(2:end,5)) y];

%add in the FRED data where RER is last, US GDP and oil price first
Y = y;
%Y= [100*diff(log(rawdata.FRED(:,1))) 100*log(rawdata.FRED(2:end,5)) y];
%

 Series = {'US GDP','Real Oil Price','euro area GDP','IP','Employment','Housing Permits','CPI',...
     'Policy Rate','Hours Worked','Term Spread','CAPU','Unemployment','PMI','Risk Spread','RER', ... 
     'Nonfin-Total financial assets',	'Nonfin-Total financial liabilities',	'Nonfin-Loans',...
     'Household-Total financial assets',	'Household-Total financial liabilities',	'Household-Loans',...
     'GVT-Total financial assets',	'GVT-Total financial liabilities',	'GVT-Debt securities', ...
     'TotalEconomy-Total financial assets',	'TotalEconomy-Total financial liability'	'Totaleconomy-Debt securities',...
     'TotalEconomy-Loans'
 };
    % Series = {'euro area GDP','IP','Employment','Housing Permits','CPI',...
    % 'Policy Rate','Hours Worked','Term Spread','CAPU','Unemployment','PMI','Risk Spread','Household Debt'};
%%
ML_graph_options 															% customized set of graphs options

giorno='20190329'; 															% vintage we will be using
gm= datenum(str2num(giorno(1:4)),str2num(giorno(5:6)),str2num(giorno(7:8))); % ------------------------
%Filename=['USDB_Haver_' num2str(gm)];										% Name of data file

%%% ========================== %%%
%%%  Set options for the model %%%
%%% ========================== %%%
tresh=10^(-2);                                                              % tolerance treshold for EM algorithm
star=10^(-5);                                                               % Initial variance R 
maxiter=50;                                                                 % max number iteratioin EM algorithm
trans=3; out=2;                                                             % data transformation
cut=0; StartYear=1999.25;                                                   % cut the sample if cut ==1
q=5; s=1; d=q-1; p=4; det=1;                                                % parameters DFM
GDO=0;                                                                      % impose GDO restrictions
model='VAR';                                                                % determines law of motion for the factors
m=0; cc=10;                                                                 % parameters Robinson-Yao-Zhang 
%TR1=[6 15 72 73 76:80 87];                                                 % Variables for which I overwrite the trend test
I0=[]; I1=[];                     					                        % Restrict some idio to be I(0) in the EM
rr=ones(q,1); 																% initialize all factors with diffuse variance
%TV.id={1:2,75};                                                            % time varying parameters
TV.id={1};
TV.Type={'trend'};                                                          % -----------------------
TV.q0=[10^(-2), 10^(-2)];                                                   % initial variance for TV states
dates = (1999:0.25:2019.75)'; %dates label

%%% =========== %%%
%%%  Load Data  %%%
%%% =========== %%%
% [Y, Label, Name, Dates, cd, CBO, idcat, NameC] = ...                        % Load data in levels    
%     ML_ReadDataHaver(Filename,trans,out,cut,StartYear,[],1);                % -------------------
% disp(['Data were downloaded on ' datestr(gm)]); disp(' ')
% 
% disp(' *** ====================== ***')
% disp(' ***  Model specification:  ***')
% disp(' *** ====================== ***')
% disp(['q=' num2str(q) ' s=' num2str(s) ' d=' num2str(d) ' p=' num2str(p)])  % --------------
% disp(['Law of motion for the Factors is: ' model]);
% if GDO==1; disp('GDO restriction is imposed'); end;disp(' '); 
% 
% if ~isempty(TV.id)
%     disp('**************************************')
% 	disp('Time-varying deterministic components:')
%     disp('--------------------------------------')
% 	for jj=1:length(TV.id)
% 		disp(['TV ' TV.Type{jj}(1,:) ' on:'])
% 		disp(char(Name(TV.id{jj})))
%         disp(['Starting variance: ' num2str(TV.q0(jj))]);
%         disp(' ')
%     end
%     disp('--------------------------------------')
% end
%%
%%% =============================== %%%
%%%  Initialize the model with PCA  %%%
%%% =============================== %%%
y=ML_diff(Y);                                                              % Data in 1st Differences
[T,N]=size(y);                                                             % size of the panel
[yy, my, sy]=ML_Standardize(y);                                            % Standardize
[f,lambda]=ML_efactors2(yy,q,2);                                           % estimate factor loadings, aka DFM on \Delta y_t
TT=(1:T+1)';                                                               % time trend
X=NaN(T+1,N); bt=X; b=zeros(N,2);                                          % preallocates variables for detrending
J=BL_TestLinearTrend(y);                                                   % Identify variables to be detrended
[X(:,J),bt(:,J),b(J,:)]=ML_detrend(Y(:,J));                                % Detrend variables to be detrended
X(:,~J)=Y(:,~J)-repmat(mean(Y(:,~J)),T+1,1);                               % Demean variables not to be detrended
bt(:,~J)=repmat(mean(Y(:,~J)),T+1,1);                                      % ---------------------
b(~J,1)=mean(Y(:,~J))';b(~J,2)=0;    
%% ---------------------
if GDO==1 % --------------------------------------------------------------- % Restrictions for GDO
    lambda(1:2,:)=repmat(mean(lambda(1:2,:)),2,1);                          % same loadings
    b(1:2,2)=mean(b(1:2,2));                                                % same slope, different constant
    bt(:,1:2)=repmat(b(1:2,1)',T,1)+TT*b(1:2,2)';                           % same linear trend
    X(:,1:2)=(y(:,1:2)-bt(:,1:2));                                          % Detrended GDP GDI
    sy(1:2)=repmat(mean(sy(1:2)),1,2);                                      % same std for standardization
end        % -------------------------------------------------------------- %
Z=X./repmat(sy,size(X,1),1);                                                % Standardize detrended variables   
F=Z*lambda/N;                                                               % Factors in levels as in BLL
[A0,v,AL]=ML_VAR(F,p,1);                            						% Estimate VAR on common factors
xi=Z-F*lambda';                                                             % idiosyncratic component
%%
%%% ============================ %%%
%%%  Estimate the model with EM  %%%
%%% ============================ %%%
Z2=(Y-repmat(b(:,1)',size(Y,1),1))./repmat(sy,size(Y,1),1);                 % BL standardizations: (Y_t - a), i.e. approx. centered levels, divided by sy 
b1=b./repmat(sy',1,2); b1(:,1)=0;                                           % divide slope by sy, constant=0;
[A,Q,Lambda,R,F00,P0s,mu,type2]=ML_DynamicFactorSS_GDO_TV...                % State-space representation
    (AL,lambda(:,1:q),v,q,s,xi,F,rr,I0,I1,b1,A0,TV,star);                   % --------------------------
[xitT,PtT,PtTm,~,~,Ptt,~,A,Lambda,R,Q,b2,mu2]=ML_DynamicFactorEM_GDO_TV...  % EM-Algorithm
    (Z2,F00,P0s,A,Lambda,R,Q,s,q,p,mu,type2,maxiter,tresh,cc,GDO);          % ------------
eta=Z2-xitT*Lambda';
T2=size(Z2,1); start=1;
BT=TT*b2';                                                                  % ML estimates of linera trend
isTV=find(type2==10)+max(p*q,q*(s+1));                                      % identifies TV coefficients
for ii=1:length(isTV) % --------------------------------------------------- % TV slopes or means
    id=find(Lambda(:,isTV(ii)));
    BT(:,id)=xitT(:,isTV(ii))*ones(1,length(id));                     
end                   % --------------------------------------------------- %
FF=xitT(:,1:q); FF1=[]; for ss=0:s;FF1=cat(2,FF1,FF(s-ss+1:end-ss,:));end   % Store ML Factors
for ss=1:s+1; lambda2(:,:,ss)=Lambda(:,(ss-1)*q+1:ss*q); end                % Store ML loadings
L=reshape(lambda2,N,q*(s+1));
%%
%%% =================== %%%
%%%  Common Components  %%%
%%% =================== %%%
T3=length(FF1(cc:end,:));
SY=repmat(sy,T3,1);
MY=repmat(b(:,1)',T3,1);
start2=start+s+cc-1; % CHECK THIS OUT, FUNDAMENTAL
Y2=Y(start2:end,:);
chi=(FF1(cc:end,:)*L'+BT(start2:end,:)).*SY+MY;                             % common component
zeta=Y2-chi;                                                                % idiosyncratic component    

%%% ================================= %%%
%%%  Estimate trend with PCA on LRCV  %%%
%%% ================================= %%%
[V, D]= eig(cov(FF(cc:end,:)));                                             % eigenvalue and eigenvenctors of VCV
[~, t2]=sort(diag(D),'descend'); V=V(:,t2); D=D(t2,t2);                     % sort eigenvalues and eigenvectors
FFt = FF(cc:end,:)*V(:,1:q-d);                                              % TREND Factors
FFc = FF(cc:end,:)*V(:,q-d+1:q);                                            % CYCLE Factors
FFt1=[]; FFc1=[];   
for ss=1:s+1     
    FFt1=cat(2,FFt1,FFt(s-ss+2:end-ss+1,:));                                % lagged trend factors
    FFc1=cat(2,FFc1,FFc(s-ss+2:end-ss+1,:));                                % lagged cycle factors
    lambdat(:,(ss-1)*(q-d)+1:ss*(q-d)) = lambda2(:,:,ss)*V(:,1:q-d);     	% trend loadings
    lambdac(:,(ss-1)*d+1:ss*d) = lambda2(:,:,ss)*V(:,q-d+1:q);           	% cycle loadings    
end

chit = (FFt1*lambdat'+BT(start2:end,:)).*SY+MY;                             % common stochastic trend plus linear trend
chist = (FFt1*lambdat').*SY+MY;                                             % common stochastic trend
chic = (FFc1*lambdac').*SY;                                                 % common cyclical component
chilt = (BT(start2:end,:)).*SY;                                             % linear trend

%%
%%% =============== %%%
%%%  Euro benchmark  %%%
%%% =============== %%%
useless= xlsread('dataset.xlsx','external_gaps','G2:G85');
useless2= xlsread('dataset.xlsx','external_gaps','H2:H85');
UCM = 100*useless; %Basically it's cboc
Wong = useless2;

 
% J=ismember(CBO(:,1),Dates);
% cbot=100*log(CBO(J,2));                                                     % Potential Output CBO
% cbost=cbot-bt(:,1);                                                         % Potential Output CBO
% cboc=y(:,1)-cbot;                                                           % Output gap CBO

%%
%%%%%  ======================================  %%%%%
%%%%%  ====  --------------------------  ====  %%%%%
%%%%%  ====  ---                    ---  ====  %%%%%
%%%%%  ====  ---  Plotting Results  ---  ====  %%%%%
%%%%%  ====  ---                    ---  ====  %%%%%
%%%%%  ====  --------------------------  ====  %%%%%
%%%%%  ======================================  %%%%%
startDate = '01-Jan-1999';  
endDate = '01-Jan-2020';   

% Generate vector of dates in datenum format
Dates = datenum(startDate):92:datenum(endDate); % 91 days is roughly a quarter
Dates= Dates';
%%
Dates2=Dates(start2:end); Dates2d=Dates2(2:end); dates4q=Dates2(5:end);    % New Dates for plots
j0= 1;                                                                     % Starting point for all graphs 
colore={[0 0.45 0.74],[0.64 0.08 0.18],[0.85 0.33 0.1],...                 % colors for gaps blue, red, orange
    [0.93 0.69 0.13],[0 0 0],[0.47 0.67 0.19],[0.49 0.18 0.56]};           % yellow,black, green, purple
qsp=[num2str(q) num2str(s) num2str(p)];
%DD=Dates2(j0:end);
DD=Dates2;
%%

%%% ======================= %%%
%%%  Gross Domestic Output  %%%
%%% ======================= %%%
ZZ=[UCM(start2:end,1) Wong(start2:end,1) chic(:,1)]; 
ZZ=ZZ(1:end,:);              				                                % Output Gap - levels
SizeXY=ML_SizeXY(DD,ZZ,1,5);                                                % -------------------
ML_TimeSeriesUS(ZZ,DD,' ',{'UCM','Wong','BL'},' ',SizeXY,0,colore);             	% -------------------
title('Output Gap - Level','fontweight','bold','fontsize',16)

yearlyTicks = unique(year(Dates));  
set(gca, 'XTick', datenum(yearlyTicks, 1, 1));  
set(gca, 'XTickLabel', yearlyTicks);  
set(gca, 'Box', 'off');  
set(gca, 'XAxisLocation', 'bottom');


lavender = [230,230,250]/255;


peak   = [2008;  2011+2/12];
trough = [2009+1/12; 2013];


hold on;


yLim = ylim;
yMin = yLim(1);
yMax = yLim(2);


fills = [];


for i = 1:length(peak)
    startDate = datenum(peak(i), 1, 1);
    endDate = datenum(trough(i), 1, 1);
    f = fill([startDate startDate endDate endDate], [yMin yMax yMax yMin], lavender, 'EdgeColor', 'none','HandleVisibility', 'off');
    fills = [fills; f];
end


for i = 1:length(fills)
    uistack(fills(i), 'bottom');
end


hold off;
%%
ZZ=[UCM(start2:end,1) Wong(start2:end,1) chic(:,1)]; ZZ=ML_diff(ZZ(j0:end,:),1);            	% Output Gap - 4Q
SizeXY=ML_SizeXY(dates4q,ZZ,1,5);  SizeXY(3:4)=[-6 6];                      % ---------------
ML_TimeSeriesUS(ZZ,dates4q(j0:end),' ',{'UCM','Wong','BL'},' ',SizeXY,0,colore);   % ---------------
title('Output Gap - 4-quarter percent changes','fontweight','bold','fontsize',16) % ---------------

yearlyTicks = unique(year(Dates)); 
set(gca, 'XTick', datenum(yearlyTicks, 1, 1));  
set(gca, 'XTickLabel', yearlyTicks);  
set(gca, 'Box', 'off');  
set(gca, 'XAxisLocation', 'bottom');
lavender = [230,230,250]/255;
peak   = [2008;  2011+2/12];
trough = [2009+1/12; 2013];


hold on;


yLim = ylim;
yMin = yLim(1);
yMax = yLim(2);


fills = [];


for i = 1:length(peak)
    startDate = datenum(peak(i), 1, 1);
    endDate = datenum(trough(i), 1, 1);
    f = fill([startDate startDate endDate endDate], [yMin yMax yMax yMin], lavender, 'EdgeColor', 'none','HandleVisibility', 'off');
    fills = [fills; f];
end

for i = 1:length(fills)
    uistack(fills(i), 'bottom');
end

hold off;