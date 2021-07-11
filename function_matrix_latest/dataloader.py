# Copyright (c) 2021 Dudu 版权所有
#
# 文件名：dataloader.py
# 功能描述：线性模型求解功效矩阵
#
# 作者：Dudu
# 时间：7月11日
#
# 版本：V1.0.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MatrixDataLoader:
    """
    读取csv数据
    csv_list: [path1, path2...] list
    x_cols: x data column index
    y_cols: y data column index
    """
    def __init__(self, csv_list, x_cols: list, y_cols: list):
        self.csv_list = csv_list
        self.alldata = self.read_data(csv_list)

        self.x_cols = x_cols
        self.y_cols = y_cols

    def read_data(self, csv_list):
        """
        从csv——list读取所有数据
        :param csv_list: csv file list
        :return: pandas.DataFrame
        """
        data = []
        data_null_check = []
        for csv_path in csv_list:
            d = pd.read_csv(csv_path)
            data_null_check.append(d)
            d_ = d.shift(1)
            d = d.iloc[:, :] - d_.iloc[:, :]
            d = d.iloc[1:, :]
            data.append(d)
        d = pd.concat(data)
        data_null_check = pd.concat(data_null_check)
        self.analyse_data(data_null_check)
        print('文件数据读取成功，CSV数据维度：', data_null_check.shape, '训练数据：', d.shape)
        print(d.head(5))
        return d

    def analyse_data(self, data):
        """
        分析所有数据
        :param data: dataframe
        :return: None
        """
        assert isinstance(data, pd.DataFrame), '数据类型错误，必须为pandas.DataFrame类型'
        print(data.isnull().sum())
        return

    def __call__(self, limit=None):
        """
        返回x_cols 和 y_cols的数据，注意，这是训练数据，要减去上一行
        :param: limit [[lim1, lim2]...]
        :return:data np.ndarry
        """
        x_data = self.alldata.iloc[:, self.x_cols].to_numpy()
        y_data = self.alldata.iloc[:, self.y_cols].to_numpy()
        if limit is None:
            return x_data, y_data
        # 根据值过滤数据
        assert len(limit) == x_data.shape[1], 'x列数和限制列表必须相同'
        length = x_data.shape[1]
        x_filter, y_filter = [], []
        for x, y in zip(x_data, y_data):
            flag = True
            for i in range(length):
                if x[i] < limit[i][0] or x[i] > limit[i][1]:
                    flag = False
            if flag:
                x_filter.append(x)
                y_filter.append(y)
        x_data = np.vstack(x_filter)
        y_data = np.vstack(y_filter)
        return x_data, y_data



if __name__ == '__main__':
    files = ['848543.csv', '848794.csv', '848916.csv', '849094.csv', '849205.csv']
    y_cols = [i for i in range(36, 70)]
    x_cols = [4, 9, 10]
    dataloader = MatrixDataLoader(files, x_cols, y_cols)

    # 绘制x直方图，源数据
    x, y = dataloader()
    for i in range(len(x_cols)):
        d = x[:, i]
        print('数据最大最小值为：', d.max(), d.min())
        plt.hist(d, bins = 100)
        plt.show()

    # 绘制x直方图，限制后的数据
    x_limits = [[-20, 20], [-5, 5], [-3, 3]]
    x, y = dataloader(limit=x_limits)
    print('限制后数据维度：', x.shape, y.shape)
    for i in range(len(x_cols)):
        d = x[:, i]
        print('数据最大最小值为：', d.max(), d.min())
        plt.hist(d, bins=50)
        plt.show()


