function NormalizedMatrix = DiagonalSignNorm(matrix)
    for i = 1:size(matrix, 1)
        if matrix(i, i) < 0
            matrix(:, i) = -matrix(:, i);
        end
    end
    NormalizedMatrix = matrix;
end
%A simple loop that given a matrix restricts all its diagonal elements to
%be positive. Morevor if a diagonal element is found to be negative it
%will also invert the sign of all elements in that column.