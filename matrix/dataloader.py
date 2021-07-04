# Copyright (c) 2021 Dudu 版权所有
#
# 文件名：dataloader.py
# 功能描述：功效矩阵求解数据读取
#
# 作者：Dudu
# 时间：2021年6月24日
#
# 版本：V1.0.0

import numpy as np
import csv

class CsvDataloader:
    """
    读取csv数据
    file_list:list，csv文件列表
    col_index_list:list, 数据列索引列表
    """
    def __init__(self, file_list, train_x_cols=None, train_y_cols=None, x_threshold=[-30, 30], y_threshold=[-20, 20]):
        self.file_list = file_list
        self.train_x_cols = train_x_cols
        self.train_y_cols = train_y_cols
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold

        self.x, self.y = self.get_data_by_col_list()

    def __len__(self):
        return self.x.shape[0]

    def read_csv_data(self, file_paths: list):
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
                    if idx != 0 and '0' not in row:
                        data.append(row)
            data = np.array(data, dtype='float')
            data_pre = data[0:-1, :]
            data_aft = data[1:, :]
            data = data_aft - data_pre
            alldata.append(data)
        alldata = np.vstack(alldata)
        return alldata

    def get_data_by_col_list(self):
        data = self.read_csv_data(self.file_list)
        x, y = data[:, self.train_x_cols], data[:, self.train_y_cols]
        x, y = self.clear_data(x, y)
        return x, y

    def clear_data(self, x, y):
        new_x = []
        new_y = []
        for x_l, y_l in zip(x, y):
            if np.any(x_l < self.x_threshold[0]) or np.any(x_l > self.x_threshold[1]) or \
                np.any(y_l < self.y_threshold[0]) or np.any(y_l > self.y_threshold[1]):
                pass
            else:
                new_x.append(x_l)
                new_y.append(y_l)
        return np.vstack(new_x), np.vstack(new_y)

    def __call__(self):
        return self.x, self.y

if __name__ == '__main__':
    files = ['848543.csv', '848794.csv', '848916.csv', '849094.csv', '849205.csv']
    y_cols = [i for i in range(36, 70)]
    x_cols = [4, 9, 10]
    dataloader = CsvDataloader(files, x_cols, y_cols)
    train_x, train_y = dataloader()






