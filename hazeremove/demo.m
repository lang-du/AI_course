clc;
clear all;
close all;


img_path = 'frog1.png';
rgb = imread(img_path);
rgb = double(rgb) / 255;

patch = 13;
w = 0.95;
t0 = 0.1;

filter = hazeRemoval(rgb, patch, w, t0);


subplot(1, 2, 1)
imshow(rgb)

subplot(1, 2, 2)
imshow(filter)

