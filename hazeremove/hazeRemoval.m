function [filtered] = hazeRemoval(rgb, patchSize, w, t0)
%hazeRemoval: Single Image Haze Removal Using Dark Channel Prior
%   此处显示详细说明
% inputs:
%   rgb: rgb 3 channels image, [0, 1]
%   patchSize: patch size
%   w: emission coefficient
%   t0: minimum t
% outputs:
%   filtered: rgb 3 channels image, [0, 1]

[dark] = darkDetection(rgb, patchSize);
A = getAtmosphericLight(rgb, dark, 0.1);
T = getMediumTransmission(rgb, A, patchSize, w, t0);
[T_] = generateLaplacian(rgb,T(:, :, 1));
T_ = cat(3, T_, T_, T_);
filtered = (rgb - A) ./ T_ + A;
filtered(filtered < 0) = 0;
filtered(filtered > 1) = 1;
end

