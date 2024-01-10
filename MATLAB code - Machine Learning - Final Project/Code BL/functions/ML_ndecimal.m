% ML_ndecimal - counts the number of decimals in x

% Matteo Luciani (matteoluciani@yahoo.it)

function y = ML_ndecimal(x)
x = abs(x);                 % in case of negative numbers
y = 0;
while (floor(x)~=x)
    y = y+1;
    x = x*10;
end