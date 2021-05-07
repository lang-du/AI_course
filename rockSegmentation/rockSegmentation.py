# Copyright (c) 2021 郎督 版权所有
#
# 文件名：rockSegmentation.py
# 功能描述：根据训练数据训练svc模型，对新数据分类
#
# 作者：郎督
# 时间：2021年5月6日
#
# 版本：V1.0.0

import cv2
import numpy as np
from sklearn import svm


def filter_area(image, area_threshold=200):
    """
    过滤image前景中面积小于area_threshold的部分
    :param image: 二值图，uint8
    :param area_threshold: 面积阈值，小于该值的面积被过滤
    :return: mask
    """
    assert len(image.shape)==2, "filter_area函数只能过滤二值图"
    assert isinstance(area_threshold, int), "area_threshold 必须为整数"
    mask = np.zeros(image.shape, dtype='uint8')
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(1, labels.max() + 1):
        if stats[i, -1] > area_threshold:
            mask[labels == i] = 255
    return mask


def norm(matrix):
    """矩阵归一化"""
    assert len(matrix.shape) == 2, '必须为2维矩阵'
    max = matrix.max()
    min = matrix.min()
    norm_matrix = (matrix - min) / (max - min)
    return norm_matrix

def extracting(img, h, w):
    """
    根据输入的图片以及窗口大小提取特征
    :param img: 二维uint8矩阵，维度(img_h, img_w)
    :param h: 窗口高度
    :param w: 窗口宽度
    :return: 特征，维度(img_h, img_w, number_features)
    """
    img_h, img_w = img.shape
    assert w%2==1 and h%2==1, "窗口必须为奇数"

    # 边界扩充，边缘镜像填充
    img = cv2.copyMakeBorder(img, h//2, h//2, w//2, w//2, cv2.BORDER_REFLECT)

    # 特征1：(maximum - minimum)/mean，区域极大值减去极小值/均值
    sub_feature = np.zeros((img_h, img_w), dtype='float32')
    # 特征2：std，区域标准差
    std_feature = np.zeros((img_h, img_w), dtype='float32')
    # 特征3：abs(pix_value - mean)/mean，区域平均值减去区域中心点像素值的绝对值/均值
    psm_feature = np.zeros((img_h, img_w), dtype='float32')
    # 特征4：(maximum - mean)/mean，区域极大值减去区域均值的绝对值/均值
    masm_feature = np.zeros((img_h, img_w), dtype='float32')
    # 特征5：(mean - minimum)/mean，区域均值减去区域极小值的绝对值/均值
    msmi_feature = np.zeros((img_h, img_w), dtype='float32')
    for row in range(h//2, img_h + h//2):
        for col in range(w//2, img_w + w//2):
            local_map = img[row - h//2:row + h//2 + 1, col - w//2:col + w//2 + 1]

            local_max = local_map.max() # 区域极大值
            local_min = local_map.min() # 区域极小值
            local_mean = local_map.mean()   # 区域均值
            pixel = img[row][col]       # 区域中心像素值
            # ----特征1-------
            sub_feature[row - h//2][col - w//2] = (local_max - local_min) / local_mean

            # ----特征2-------
            std_feature[row - h//2][col - w//2] = np.std(local_map)

            # ----特征3-------
            psm_feature[row - h // 2][col - w // 2] = np.abs(pixel - local_mean) / local_mean

            # ----特征4-------
            masm_feature[row - h // 2][col - w // 2] = np.abs(local_max - local_mean) / local_mean

            # ----特征5-------
            msmi_feature[row - h // 2][col - w // 2] = np.abs(local_min - local_mean) / local_mean

    # 特征图1归一化
    sub_feature = norm(sub_feature)
    # cv2.imshow('sub feature', sub_feature)
    # sub_feature = cv2.normalize(sub_feature, sub_feature, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # cv2.imwrite('sub_feature.png', sub_feature)
    # 特征2归一化
    std_feature = norm(std_feature)
    # cv2.imshow('std feature', std_feature)
    # std_feature = cv2.normalize(std_feature, std_feature, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # cv2.imwrite('std_feature.png', std_feature)
    # 特征3归一化
    psm_feature = norm(psm_feature)
    # cv2.imshow('psm_feature', psm_feature)
    # psm_feature = cv2.normalize(psm_feature, psm_feature, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # cv2.imwrite('psm_feature.png', psm_feature)
    # 特征4归一化
    masm_feature = norm(masm_feature)
    # cv2.imshow('masm_feature', masm_feature)
    # masm_feature = cv2.normalize(masm_feature, masm_feature, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # cv2.imwrite('masm_feature.png', masm_feature)
    # 特征5归一化
    msmi_feature = norm(msmi_feature)
    # cv2.imshow('msmi_feature', msmi_feature)
    # msmi_feature = cv2.normalize(msmi_feature, msmi_feature, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # cv2.imwrite('msmi_feature.png', msmi_feature)
    # print(msmi_feature.dtype)
    # 特征整合
    feature = np.stack([sub_feature, std_feature, psm_feature, masm_feature, msmi_feature], axis=-1)
    return feature

def get_data_and_label(data_path_list, label_path_list):
    """
    根据路径读取数据和标签
    :param data_path_list: [img_path1, img_path2...], type=list
    :param label_path_list: [label_path1, label_path2...], type=list
    :return:
        data:np.ndarray，uint8灰度图
        label:np.ndarray,uint8灰度图，值只有0和1，0表示背景，1表示石头
    """
    assert len(data_path_list) == len(label_path_list), "图片数量和标签数量必须相同"
    data = []
    label = []
    for data_path, label_path in zip(data_path_list, label_path_list):
        # 读取图片
        img = cv2.imread(data_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 读取标签
        lab = cv2.imread(label_path)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        lab[lab > 0] = 1        # 将石头区域标记位1，背景为0

        assert gray_img.shape == lab.shape, '图片与标签不匹配'
        data.append(gray_img)
        label.append(lab)
    data = np.hstack(data)
    label = np.hstack(label)
    # cv2.imshow('data', data)
    # cv2.imshow('label', label.astype('float'))
    return data, label


def display_mask_on_gray_img(gray_img, mask, color='blue'):
    """
    根据mask，在img对于区域显示颜色
    :param color_weight: 显示颜色的比重
    :param gray_img: 灰度图
    :param mask: 值只有0和255的图
    :param color: 颜色bgr
    :return: bgr3通道图
    """
    assert gray_img.shape == mask.shape, '图片和掩膜维度必须相同'
    if color=='blue':
        c = 0
    elif color == 'green':
        c = 1
    else:
        c = 2
    img_bgr = np.stack([gray_img, gray_img, gray_img], axis=-1)
    img_bgr[mask>0, c] = 255
    return img_bgr

def main():
    # 使用图1和图3训练，图二预测
    img_path_list = ['images/1.bmp', 'images/3.bmp']
    label_path_list = ['images/1.png', 'images/3.png']
    pred_img_path_list = ['images/2.bmp']   # 图1、2、3是题目给的图，4、5是另找的图
    pred_label_path_list = pred_img_path_list     # 预测图片，没有标签

    # -----------------------------训练阶段--------------------------
    # 读取训练数据
    data, label = get_data_and_label(img_path_list, label_path_list)
    # cv2.imshow('data', data)        # 显示中，灰度图模式

    # ###########第一问：提取特征##################
    h = 19  # 窗口高度
    w = 19  # 窗口宽度
    feature = extracting(data, h, w)

    # reshape,以便送入svm训练
    feature = np.reshape(feature, (-1, feature.shape[-1]))
    label = np.reshape(label, (feature.shape[0]))

    # #############第二问：svm分类#################
    print('svm训练中，训练数据维度：', feature.shape, '请等待---------------')
    clf = svm.SVC(
        C=1,
        kernel='sigmoid',   # rbf, poly, sigmoid, linear, precomputed
        degree=3,
        max_iter=300
    )
    clf.fit(feature, label)

    # --------------------------svm预测-----------------------------------
    # 读取测试数据
    test_data, test_label = get_data_and_label(pred_img_path_list, pred_label_path_list)

    # 提取特征
    test_feature = extracting(test_data, h, w)

    # reshape,并送入svm预测
    test_feature = np.reshape(test_feature, (-1, test_feature.shape[-1]))
    pred_label = clf.predict(test_feature)

    # 恢复维度
    pred_label = np.reshape(pred_label, (test_label.shape))

    # 查看分割结果
    pred_label = pred_label.astype('uint8')

    pred_label[pred_label == 1] = 255   # mask
    display_img = display_mask_on_gray_img(test_data, pred_label)
    cv2.imshow('segmented img', display_img)
    # cv2.imwrite('display_img.png', display_img)

    # --------------------第三问：大岩石表面误判----------------------
    # 计算检测的平坦区域面积，过滤面积较小区域
    flat_area = 255 - pred_label       # 过滤小的平坦区域，将前景背景置换，平坦区域为前景
    flat_area = filter_area(flat_area, area_threshold=400)  # 过滤小面积平坦区域
    rock_area = 255 - flat_area         # 前景背景置换，平坦区域为背景
    filter_img = display_mask_on_gray_img(test_data, rock_area, color='blue')
    cv2.imshow('filted img', filter_img)
    # cv2.imwrite('filter_new_img.png', filter_img)

    # ----------第四问：不同地形-----------
    # 更换预测的图片路径，即pred_img_path_list，可以更换为图4或者图5

    cv2.waitKey(0)

if __name__ == '__main__':
    main()



