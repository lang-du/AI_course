function [EP_] = updateEP(EP, F)
%updateEP 更新EP，移除其中被F支配的向量
%   此处显示详细说明
%   F: 1x2
%   EP: Nx2
EP_ = [];

s = size(EP);
if s(1) == 0
    EP_ = [EP_; F];
end
% 移除被F支配的向量
flag = false;   % F是否被支配
if s(1) > 0
    for i = 1:s(1)
        if EP(i, 1) >= F(1) && EP(i, 2) >= F(2)   % 被F支配
            continue
        elseif EP(i, 1) <= F(1) && EP(i, 2) <= F(2)   % F被支配
            flag = true;
        end
        EP_ = [EP_; EP(i, :)];
    end
end
if flag == false
    EP_ = [EP_; F];
end
end

