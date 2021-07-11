# Copyright (c) 2021 Dudu 版权所有
#
# 文件名：linearModel.py
# 功能描述：线性模型求解功效矩阵
#
# 作者：Dudu
# 时间：2021年7月11日
#
# 版本：V1.0.0

from dataloader import MatrixDataLoader
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


def plot_functional_coef(coef, legend_list=None, marker_list=None, title='matrix'):
    """
    绘制功效矩阵图像
    :param coef: matrix
    :return:None
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if legend_list:
        assert len(legend_list) == coef.shape[1], 'lengend长度必须和矩阵第二维度相同'
        assert len(legend_list) == len(marker_list), 'legend_list和marker_list长度必须相同'
    plt_handles = []
    for i in range(len(legend_list)):
        x_range = [i for i in range(coef.shape[0])]
        p,  = plt.plot(x_range, coef[:, i], marker=marker_list[i], label=legend_list[i])
        plt_handles.append(p)
    plt.legend(handles=plt_handles)
    plt.title(title)
    plt.show()


def main():
    files = ['848543.csv', '848794.csv', '849094.csv', '849205.csv', '848916.csv',]
    y_cols = [i for i in range(36, 70)]
    x_cols = [4, 9, 10]
    x_limits = [[-20, 20], [-5, 5], [-3, 3]]
    dataloader = MatrixDataLoader(files, x_cols, y_cols)
    x, y = dataloader(limit=x_limits)

    # 归一化
    scaler = MinMaxScaler((-1, 1))
    x = scaler.fit_transform(x)
    # y = scaler.fit_transform(y)
    print(x.max(), x.min())

    # train_x, test_x, train_y, test_y = train_test_split(x, y)
    # 最小二乘
    reg = LinearRegression()
    reg.fit(x, y)
    coef = reg.coef_
    print('线性回归功效矩阵：', coef)
    # plot functional coef
    plot_functional_coef(coef, ['轧制力', '弯辊力', '中间辊弯辊力'], ['o', '*', '>'], title='linear regression')

    # # 脊回归
    reg = Ridge(.5)
    reg.fit(x, y)

    coef = reg.coef_
    print('脊回归功效矩阵：', coef)
    plot_functional_coef(coef, ['轧制力', '弯辊力', '中间辊弯辊力'], ['o', '*', '>'], title='ridge regression')

    # ann
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    regr = MLPRegressor(
        hidden_layer_sizes=(64, ),
        activation='identity',
        max_iter=100,
        n_iter_no_change=100,
    )
    regr.fit(X_train, y_train)
    R = regr.score(X_test, y_test)  # coefficient of determination
    coef_list = regr.coefs_
    coef = (coef_list[0] @ coef_list[1]).T      # matrix
    print('ANN matrix:', coef)
    print('the coefficient of determination:', R)

    loss = regr.loss_curve_ # training loss
    plt.title('ANN loss')
    plt.plot(loss)
    plt.show()

    plot_functional_coef(coef, ['轧制力', '弯辊力', '中间辊弯辊力'], ['o', '*', '>'], title='ANN regression')




if __name__ == '__main__':
    main()


