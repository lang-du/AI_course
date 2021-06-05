function [dark] = darkDetection(rgb, patchSize)
%darkDetecton 计算暗通道先验
%   此处显示详细说明
% inputs:
%   rgb:color image
%   patchSize:window size
% outputs:
%   dark:dark image

dark_channel = min(rgb, [], 3);
dark = zeros(size(dark_channel));
[h, w, c] = size(rgb);
pad = fix(patchSize / 2);
pad_dark = padarray(dark_channel, [pad, pad], 'symmetric'); % 填充边界

for i = pad + 1:h + pad
    for j = pad + 1:w + pad
        patch = pad_dark(i - pad:i + pad, j - pad:j + pad);
        dark(i - pad, j - pad) = min(patch, [], 'all');
    end
end

end

