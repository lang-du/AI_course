# Copyright (c) 2021 郎督 版权所有
#
# 文件名：demo.py
# 功能描述：板形控制功效矩阵求解
#
# 作者：郎督
# 时间：2021年6月3日
#
# 版本：V1.0.1
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression


def split_data(data, val_split: float):
    """
    划分数据
    :param:data numpy.ndarrray
    :param:val_split split 比例
    :return: train_data, train_label, val_data, val_label
    """
    val_number = int(data.shape[0] * (1 - val_split))
    data, label = data[:, 0:-34], data[:, -34:]
    train_data, train_label = data[0:val_number], label[0:val_number]
    val_data, val_label = data[val_number:], label[val_number:]
    return train_data, train_label, val_data, val_label


def read_csv_data(file_paths: list):
    """
    读取数据，数据为每一行减去上一行的数据
    :param :file_paths  文件路径 list
    :return: data
    """
    alldata = []
    for path in file_paths:
        data = []
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(reader):
                if idx != 0:
                    data.append(row)
        data = np.array(data, dtype='float')
        data_pre = data[0:-1, :]
        data_aft = data[1:, :]
        data = data_aft - data_pre
        alldata.append(data)
    alldata = np.vstack(alldata)
    return alldata


def learning(Einit, train_data, train_label, val_data, val_label, epoches=100, lr=0.01, shuffle=True, print_epoch=5):
    """
    学习，随机梯度下降法
    Einit:初始化的功效矩阵系数
    train_data:数据
    train_label:标签
    val_data:validation data
    val_label:validation label
    epoches:次数
    lr:学习率
    shuffle: shuffle bool
    :return: E -np.ndarray   loss_list:list
    """
    E = Einit
    pred = predict(val_data, E)
    loss = mse_loss(pred, val_label)
    print( " val_loss:", loss)

    train_length = train_data.shape[0]
    loss_list = []
    for epoch in range(epoches):
        if shuffle:
            cc = list(zip(train_data, train_label))
            random.shuffle(cc)
            train_data[:], train_label[:] = zip(*cc)

        for i in range(train_length):
            # 计算预测值 34x1
            f_pred = E @ train_data[i].T
            # 计算梯度
            delt_f = np.outer((f_pred - train_label[i].T), train_data[i])  # 34x36
            delt_f = delt_f.clip(-1, 1)
            # print(E[0, 0:2], delt_f[0, 0:2])

            E = E - lr * delt_f
        pred = predict(val_data, E)
        loss = mse_loss(pred, val_label)
        loss_list.append(loss)
        if epoch % print_epoch == 0:
            print('Epoch:', epoch+1, " val_loss:", loss)
    return E, loss_list


def predict(data, E):
    """
    预测
    data: np.ndarray，计算的数据
    E:np.ndarray,功效矩阵
    """
    return (E @ data.T).T


def mse_loss(pred, label):
    """
    计算预测值和标签的mse损失函数
    pred:预测值
    label:标签
    return:平均损失
    """
    error = np.mean(np.power((pred - label), 2), 0)
    # print(error)
    return np.sum(error)


def normalize_data(data):
    """
    归一化数据
    :param: data 带归一化的数据,np.ndarray
    :return:data
    """
    mean_data = data.mean(axis=0)
    std_data = data.std(axis=0)
    data = (data - mean_data) / std_data
    return data


def main():
    # files = ['848543.csv', '848794.csv', '848916.csv', '849094.csv', '849205.csv']
    files = ['848543.csv']
    data = read_csv_data(files)
    data = normalize_data(data) # 标准化
    train_data, train_label, val_data, val_label = split_data(data, 0.2)
    print('训练集数据条目：', train_data.shape[0], '测试集数据条目：', val_data.shape[0])

    reg = LinearRegression().fit(train_data, train_label)
    E = reg.coef_
    print(type(E), E.dtype, E.shape)



    pred = reg.predict(val_data)
    print(pred[0, 0:5])
    print(val_label[0, 0:5])
    loss = mse_loss(pred, val_label)
    print(loss)

    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.show()


if __name__ == '__main__':
    main()
