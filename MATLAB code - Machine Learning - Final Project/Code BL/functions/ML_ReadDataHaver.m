% BLL_ReadData - upload and transforms US quarterly Data
%
% [Data, codeData] = BLL_ReadData(trans,out,sel)
%   trans = 0 - No transformation except logarithms
%   trans = 1 - Light transformation
%   trans = 2 - Heavy transformation
%   trans = 3 - Heavy Transformation but variables in Level
%   out   = 1 - Remove outliers
%   sel   = 1 - You are uploading the Real US DB
%   sel   = 2 - Standard Large US DB
%   sel   = 3 - Large US DB  with per capita variables
%

% Matteo Luciani (matteoluciani@yahoo.it)

function [DB, Ticker, Name, Dates, cd, CBO, cat, NameC, DB0, Dates0] = ...
    ML_ReadDataHaver(filename,trans,out,cut,StartYear,nosector,oil,price)


try isnan(nosector); catch, nosector=0; end
load([pwd '\functions\' filename]);                                         % Upload the Database
if nosector==1; J=small==1; else J=Sel==1; end                              % Selected Variables
x=Data(:,J); TR=TR(J,:)'; cat=cat(J,:)'; remove=remove(J);                	% ------------------
Name=Name(J); Ticker=Ticker(J); NameC=NameC(J);                             % ------------------

%%% ================ %%%
%%%  Eliminates NAN  %%%
%%% ================ %%%
J=find(sum(isnan(x),2)>0);
disp('NaN in these dates:')
disp(datestr(Dates(J,:),'dd-mmm-yyyy'))
    disp(['Last observation is: ' datestr(Dates(max(find(sum(~isnan(x),2)>0))))])

%%% =============================================== %%%
%%%               Transform Variables               %%%
%%% ----------------------------------------------- %%%
%%%     TR=1 ==> y_t = x_t                          %%%
%%%     TR=2 ==> y_t = \Delta x_t                   %%%
%%%     TR=3 ==> y_t = \Delta 100*log(x_t)          %%%
%%%     TR=4 ==> y_t = \Delta \Delta 100*log(x_t)   %%%
%%%     TR=5 ==> y_t = 100*log(x_t)                 %%%
%%% =============================================== %%%

try isnan(oil); TR(2,97)=3; end                                             % Options to fix some transformations
try isnan(price);  TR(2,[36 38 40 43])=3; end                               % -----------------------------------

if trans==2;        disp('Heavy Transformation');
    TR(1,:)=[]; 
elseif trans==3;    disp('Heavy Transformation but variables in Level');
    trans=1; TR(1,:)=[]; TR(TR==2)=1; TR(TR==3)=5; TR(TR==4)=3;
else
    TR(2,:)=[]; 
    if trans==0;    disp('No transformation - just take logs')
        TR(TR==2)=1; TR(TR>2)=5; 
    else            disp('Light Transformation')
    end
end                         

[T,N]=size(x);                                                              % Size of the panel
db=DataTransformation(x,T,N,TR);                                            % Transform Variables
DB=db(trans+1:T,:); Dates(1:trans)=[];
cd=zeros(N,1); cd(TR==2|TR==3)=1; cd(TR==4)=2;

%%% ============================ %%%
%%%  Cut the sample if required  %%%
%%% ============================ %%%
if cut==1;     
    J1=find(year(Dates)==StartYear,1);
    DB(1:J1,:)=[]; Dates(1:J1,:)=[]; 
end     

disp(['Sample Starts in: ' datestr(Dates(1))])
disp(['Sample ends in: ' datestr(Dates(end))])

%%% ===================== %%%
%%%  Eliminates Outliers  %%%
%%% ===================== %%%
if out==1; 
    DB = removeoutliers(DB);                                                % Remove outliers for all variables       
elseif out==2;                                                              % Special procedure when data are in levels  
    y=ML_diff(DB);                                                          % Data in 1st Differences
    J2=find(remove==1);                                                     % series that need to be screened
    for jj=J2';        
        y(:,jj)=removeoutliers(y(:,jj));                                    % remove outliers in 1st difference
        DB(:,jj)=[DB(1,jj); DB(1,jj)+cumsum(y(:,jj))];                      % reconstruct data in levels
    end
end

DB0=DB; Dates0=Dates; DB(J-trans,:)=[]; Dates(J-trans,:)=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function db=DataTransformation(x,T,N,TR)

db=NaN(T,N);                                                                % Preallocates
db(:,TR==1)=x(:,TR==1);                                                     % Transform Variables
db(2:T,TR==2)=ML_diff(x(:,TR==2));                                          % -------------------
db(2:T,TR==3)=ML_diff(100*log(x(:,TR==3)));                                 % -------------------
db(3:T,TR==4)=ML_diff(ML_diff(100*log(x(:,TR==4))));                        % -------------------
db(:,TR==5)=100*log(x(:,TR==5));                                            % -------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% ========================= %%%
            %%% == ------------------- == %%%
            %%% == - Remove Outliers - == %%%
            %%% == - By Mario Forni  - == %%%
            %%% == ------------------- == %%%
            %%% ========================= %%%
            
function [cleaneddata, out]= removeoutliers(data)

cleaneddata = data;
kk=0;
for i = 1:size(data,2);
    J=~isnan(data(:,i)); 
    x=(data(J,i));
    iqr_x = iqr(x);                                                         % Interquantile Range
    amd_tx = abs(x - median(x));                                            % Distance from Median Value
    a  = find(amd_tx >= 6*iqr_x);                                           % Identify Outliers    
    if ~isempty(a)
        for j = 1:length(a);
            kk=kk+1; out(kk,:)=[i a(j)];
            if a(j)>1; x(a(j)) = median(x(max(1,a(j)-5):a(j)-1));
            else         x(a(j)) =  median(x(2:6)); end
        end        
    end
    cleaneddata(J,i) = x;
end
disp(['Number of Detected Outliers: ' num2str(kk)]);
% disp(out):



%     if ~isempty(TR0);
%         db20=DataTransformation(x,T,N,TR0(1,:)); 
%         db21=db20(3:T,:);
%         db22=db21;
%         
%         cd2=zeros(N,1); cd2(TR0==2|TR0==3)=1; cd2(TR0==4)=2;
%         J=find(ML_diff(TR0)==0);
%         [db22(:,J), out] = removeoutliers(db21(:,J));
%         
%         DB2=[DB(1,:);repmat(DB(1,:),T-2,1)+cumsum(db22)];
%     else
%     end