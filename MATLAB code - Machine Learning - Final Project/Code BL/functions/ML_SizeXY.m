% ML_SizeXY (Dates,x,nd,mof) - Set size of axis for a graph
% 
% SizeXY=ML_SizeXY(Dates,x,nd,mof)
%   SizeXY = [xmin xmax ymin ymax]
%       nd = number of decimals
%      mof = rounds towards the closest multiple of
% 

% Matteo Luciani (matteoluciani@yahoo.it)

function SizeXY=ML_SizeXY(Dates,x,nd,mof)

z=10^nd;   
try w=mof; catch; w=1; end

ay=ML_min(x); ay=w*floor(z*ay/w)/z;

by=ML_min(x,2); by=w*ceil(z*by/w)/z;

SizeXY=[Dates([1 end])' ay by]; 