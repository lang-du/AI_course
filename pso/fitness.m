function [f] = fitness(F1,F2)
%fitness 适应度计算
%   此处显示详细说明

f = [F1 F2];

f = sum(f, 2);

end

