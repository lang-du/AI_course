function [F1, F2] = zdt1(X)
%zdt1 函数
%   此处显示详细说明
%   X：输入矩阵
%   F1：F1
%   F2：F2
s = size(X);
N = s(end);
F1 = X(:, 1);
g = 1 + 9 * sum(X(:, 2:N), 2) / (N - 1);
h = 1 - sqrt(F1 ./ g);
F2 = g .* h;

end

