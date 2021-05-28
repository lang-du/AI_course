% # Copyright (c) 2021 郎督 版权所有
% #
% # 文件名：PSO_zdt3.py
% # 功能描述：粒子群算法求解ZDT3，非多目标优化，优化目标为：min(F1+F2)
% #
% # 作者：郎督
% # 时间：2021年5月28日
% #
% # 版本：V1.0.0
% # 问题及bug反馈Q：1207540056

% PSO算法求解zdt3
% 参考：https://blog.csdn.net/daaikuaichuan/article/details/81382794

clc;
close all;
clear all;

p_num = 100;    % 粒子数量
iters =1000;   % 迭代次数
M = 5;  % 输入向量元素个数
w_init = 0.9;    % 速度权重初始值
w_end = 0.4;    % 速度权重end
c1 = 1;   % 速度加权系数
c2 = 1;   % 速度加权系数

v_max = 0.2;  % 粒子最大速度
v_min = -0.2; % 粒子最小速度

p_max = 2;  % 粒子最大值
p_min = 0;  % 粒子最小值


pos = rand(p_num, M);   % 初始化粒子
posV = zeros(size(pos));    % 初始化粒子速度
pbest = rand(p_num, M);            % 记录粒子最优位置

[F1, F2] = zdt3(pbest);   % 计算每个粒子的zdt值
Fbest = fitness(F1, F2);        % 最优粒子对应的适应度值，越小越好
[minF, index] = min(Fbest); % 获得最优粒子索引,以及对应的适应度
gbest = pos(index, :);  % 全局最优粒子
Fgbest = Fbest(index);  % 最优粒子适应度

% 终止条件为迭代结束
for i=1:iters
    % 更新粒子速度
    r1 = rand(1, M);
    r2 = rand(1, M);
    w = (w_init - w_end) * (iters - i) / iters + w_end;
    posV = w .* posV + c1 .* r1 .* (pbest - pos)  + c2 .* r2 .* (gbest - pos);
    
    posV(posV > v_max) = v_max; % 速度限制
    posV(posV < v_min) = v_min; % 速度限制
    
    pos = pos + posV;   % 更新粒子位置
    
    pos(pos > p_max) = p_max;
    pos(pos < p_min) = p_min;
    
    % 评估每个粒子函数适应值
    [F1, F2] = zdt3(pos);
    F = fitness(F1, F2);
    
    % 更新每个粒子历史最优位置
    [Fbest, pbest] = update(F, Fbest, pos, pbest);
    
    % 更新全局最优粒子
    [minF, index] = min(Fbest); % 获得最优粒子索引,以及对应的适应度
    if minF < Fgbest
        gbest = pos(index, :);  % 全局最优粒子
        Fgbest = minF;
    end
    
end

[F1, F2] = zdt3(pbest);
plot(F1,F2,'*b');                 %作图
% axis([0,1,-1,1]);
xlabel('F_1');ylabel('F_2');title('ZDT3')





