function [A] = getAtmosphericLight(rgb, dark, percent)
%getAtmosphericLight 获取atmospheric light
%   此处显示详细说明
% inputs:
%   rgb:rgb image
%   dark:dark prior
%   percent:pixel percentage
% outputs:
%   A:3 channels images每个通道值相同

% 统计dark概率分布
percent = percent / 100;
dark_vector = reshape(dark, [], 1);
t = tabulate(dark_vector(:));
A = rgb;
[f, xi] = ksdensity(dark_vector, t(:, 1), 'Function', 'cdf');
[length, ~] = size(f);
brightness = xi(length);    % 初始化为最高亮度
for i = length:1
    if f(i) > (1 - percent )
        brightness = xi(i);
    else
        break;
    end
end

mask = (dark >= brightness);
mask = cat(3, mask, mask, mask);

A(mask == 0) = 0;
maximum = max(A, [], [1, 2]);
A(:, :, 1) = maximum(1);
A(:, :, 2) = maximum(2);
A(:, :, 3) = maximum(3);

end

