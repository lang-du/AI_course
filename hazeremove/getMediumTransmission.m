function [T] = getMediumTransmission(rgb, A, patchSize, wCoef, t0)
%getMediumTransmission 计算medium transmission
%   此处显示详细说明
% inputs:
%   rgb: rgb 3 channels image
%   A:atmospheric light
%   patchSize:patch size
%   wCoef:constant parameter to keep haze
%   t0:lower boundary of t
% outputs:
%   T:medium transmission matrix
[h, w, c] = size(rgb);
T = zeros(h, w);

temp = rgb ./ A;
temp = min(temp, [], 3);


% region minimum
pad = fix(patchSize / 2);
pad_temp = padarray(temp, [pad, pad], 'symmetric'); % 填充边界


for i = pad + 1:h + pad
    for j = pad + 1:w + pad
        patch = pad_temp(i - pad:i + pad, j - pad:j + pad);
        t = min(patch, [], 'all');
        T(i - pad, j - pad) = max([t, t0]);
    end
end
T = 1 - wCoef * T;

end

