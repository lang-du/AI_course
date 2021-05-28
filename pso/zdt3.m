function [F1, F2] = zdt3(X)
%zdt3 此处显示有关此函数的摘要
%   此处显示详细说明
%   X:input matrix
%   F1:f1
%   F2:f2

s = size(X);
N = s(end);
F1 = X(:, 1);

g = 1 + 9 * sum(X(:, 2:N), 2) / (N - 1);
h = 1 - sqrt(F1 ./ g) - (F1 ./ g) .* sin(10 * pi .* F1);
F2 = g .* h;


end

