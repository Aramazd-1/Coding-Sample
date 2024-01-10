% ML_TimeSeriesUS - Plot US time series data with NBER Recessions
% 
% ML_TimeSeriesUS(ZZ,Dates,Titolo,legenda,asseX,sizeXY,ng,colore,ny,mq,ypace)
% 
%      ZZ = Variables to be plotted
%   Dates = Dates upon which ZZ has to be plotted (lenght(ZZ)==length(Dates)
%  Titolo = Title of the graph (not optional but can be empty)
% legenda = Entries for legend (not optional but can be empty)
%   asseX = xlabel (not optional but can be empty)
%  sizeXY = size of axis (optional)
%      ng = if ng=1 ==> no grid (optional)
%  colore = colors for the graph (optional)
%      ny = interval (in years) for grid on x-axis (optional)
%      mq = set in which month the grid of the x-axis is put (optional)
%   ypace = Pace for y-ticks (optional)

% Matteo Luciani (matteoluciani@yahoo.it)

function pl=ML_TimeSeriesUS(ZZ,Dates,Titolo,legenda,asseX,sizeXY,ng,colore,ny,mq,ypace)

%%% ===================================== %%%
%%%  Check inputs and set default values  %%%
%%% ===================================== %%%
if ~isempty(legenda); lg=1; else lg=0; end
if ~isempty(Titolo);  tl=1; else tl=0; end
if ~isempty(asseX);   aX=1; else aX=0; end
try isnan(ng); if isempty(ng); ng=0; end; catch; ng=0; end
try isnan(sizeXY); catch; sizeXY=[]; end
try iscell(colore); 
catch, colore={'k','r','b','m','c','y',...
        'k--','r--','b--','m--','c--','y--',...
        'k:','r:','b:','m:','c:','y:'}; 
end
try isnumeric(ny); if isempty(ny); ny=5; end;  catch; ny=5; end
try isnan(mq); if isempty(mq); mq=3; end; catch; mq=3; end
try isnan(ypace); ndec=['%.' num2str(ML_ndecimal(ypace)) 'f'];
catch, ypace=1; ndec='%.0f';
end

[T,N]=size(ZZ);    

grey=[.8 .8 .8];                                                            % grey is for grid
lavender=[230,230,250]/255;                                                 % lavender is for NBER Recessions

temp=char(Titolo);                                                          % if the title is too long
if length(temp)>50; Titolo=temp(1:50); else Titolo=temp; end                % ------------------------
if isempty(sizeXY);                                                         % if size of figure not provided
    xlim=Dates([1 end])'; ylim=[ML_min(ZZ) ML_min(ZZ,2)];                   % ------------------------------
else xlim=sizeXY(1:2); ylim=sizeXY(3:4);                                    % size of figure provided by User
end

if sum(sign(ylim))==0; % -------------------------------------------------- % Set-up yticks
    t1=0:-ypace:ML_round(ylim(1),0); t2=0:ypace:ML_round(ylim(2),0);
    ytick=sort([t1(2:end) t2]); 
elseif sum(sign(ylim))>0;
    ytick=ML_round(ylim(1),0):ypace:ML_round(ylim(2),0);       
else ytick=ML_round(ylim(1),0):ypace:ML_round(ylim(2),0);
end

temp=datevec(Dates); TICK=find(temp(:,2)==mq);                              % Tick at each mq of the year
TICK2=TICK(mod(temp(TICK,1),ny)==0);                                        % grid every ny years
xtl=cellstr(repmat(' ',length(TICK),1));                                    % define the stile of the ticklabel
xtl(mod(temp(TICK,1),ny)==0)=cellstr(datestr(Dates(TICK2),'yyyy'));         % ---------------------------------

load NBER_Recessions; NBER2=NBER(:,2)*ylim;

        %%% ======================= %%%
        %%%         Graphing        %%%
        %%% ======================= %%%
        
axes('Parent',figure,'FontSize',10); ML_FigureSize,hold on;
ha=area(NBER(:,1),[NBER2(:,1) NBER2(:,2)-NBER2(:,1)],'linestyle','none');  	% NBER Recessions
set(ha(1), 'FaceColor', 'none'); set(ha(2), 'FaceColor', lavender)          % ---------------
axis([xlim ylim])                                                           % define size of the figure
set(gca,'Xtick', Dates(TICK),'Xticklabel',xtl);                             % set xtick
set(gca,'Ytick',ytick,'Yticklabel',num2str(ytick',ndec));                   % set ytick
if ~ng; gridxy2(Dates(TICK2),ytick,'color',grey,'linewidth',1); end         % grid 
plot(Dates,zeros(T,1),'k','linewidth',.5);                                  % line at zero
for nn=1:N; % ------------------------------------------------------------- % plot each variable separately
    pl(nn)=plot(Dates,ZZ(:,nn),'color',colore{nn},'linewidth',1.5);         %
end         % ------------------------------------------------------------- %
hold off; box on; axis tight;
axis([xlim ylim]);                                                          % rescale the figure
J1=sum(isnan(ZZ)); J2=J1==T; pl(find(J2))=[];  
if lg; legend(pl,legenda,'location','best'); end             % legend
if tl; title(Titolo,'fontsize',18,'fontweight','bold'); end   % title
if aX; xlabel(asseX,'fontsize',14,'fontweight','bold'); end    % xlabel
set(gca,'Layer','top');
