function [Fbest, pbest] = update(F, Fbest, pos, pbest)
%update 根据适应度函数计算历史最优位置
%   此处显示详细说明
%   F：pos对应的适应度
%   Fbest：历史最优粒子对应适应度
%   pos：当前位置
%   pbest：历史最优位置

s = size(F);
length = s(1);
for i = 1:length
    if F(i) < Fbest(i)
        pbest(i, :) = pos(i, :);
        Fbest(i) = F(i);
    end
end

end

