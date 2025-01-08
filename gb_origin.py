# _*_coding:utf-8 _*_
# @Time    :2020//19 10:35
# @Author  :mrgui
# @FileName: plot_粒球分裂过程.py
# @Software: PyCharm

import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import k_means
import numpy
import matplotlib.pyplot as plt
import scipy.io
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def distances(data, p):
    # print(data, p)
    return ((data - p) ** 2).sum(axis=0) ** 0.5

# 计算标签和纯度。计算粒球中占比最大的数据
def get_label_and_purity(data):
    num = data.shape[0]
    num_positive = sum(data[:, 0] == 1)  # 正类数
    num_negative = sum(data[:, 0] == 0)  # 负类数
    purity = max(num_positive, num_negative) / num if num else 1.0
    if num_positive >= num_negative:
        label = 1
    else:
        label = 0
    return label, purity


def split_ball(data, splitting_method):
    # 临时去除数据标签
    data_no_label = data[:, 1:]
    k = len(numpy.unique(data[:, 0]))
    if splitting_method == 'k-means':
        # X: 数据; n_clusters: K的值; random_state: 随机状态（为了保证程序每次运行都分割一样的训练集和测试集）
        # 初始中心选取默认采用Kmeans++, 选取思想是聚类中心互相离得越远越好
        label_cluster = k_means(X=data_no_label, n_clusters=k, random_state=5)[1]  # 返回划分后的聚类标签，顺序和原始输入数据顺序一致
    elif splitting_method == 'center_split':
        # 采用正、负类中心直接划分
        p_left = data[data[:, 0] == 1, 1:].mean(0)
        p_right = data[data[:, 0] == 0, 1:].mean(0)
        distances_to_p_left = distances(data_no_label, p_left)
        distances_to_p_right = distances(data_no_label, p_right)
        relative_distances = distances_to_p_left - distances_to_p_right
        label_cluster = numpy.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))
    elif splitting_method == 'center_means':
        # 采用正负类中心作为 2-means 的初始中心点
        p_left = data[data[:, 0] == 1, 1:].mean(0)
        p_right = data[data[:, 0] == 0, 1:].mean(0)
        centers = numpy.vstack([p_left, p_right])
        label_cluster = k_means(X=data_no_label, n_clusters=2, init=centers, n_init=1)[1]
    else:
        return data
    # 根据聚类标签，将原始输入数据划分为两簇，即为两个粒球
    ball1 = data[label_cluster == 0, :]
    ball2 = data[label_cluster == 1, :]
    return [ball1, ball2]

def splits(granular_ball_list, purity, splitting_method):
    # 粒球list, 纯度阈值, 划分方法
    granular_ball_list_new = []
    # 遍历所有粒球
    for granular_ball in granular_ball_list:

        # 获取这个粒球的标签和纯度
        label, p = get_label_and_purity(granular_ball)

        # 如果该粒球的纯度大于纯度阈值，则不再划分
        if p >= purity:
            granular_ball_list_new.append(granular_ball)
        else:
            granular_ball_list_new.extend(split_ball(granular_ball, splitting_method))

    return granular_ball_list_new


# 计算粒球中心和半径
def calculate_center_and_radius(granular_ball):
    # print(granular_ball)
    data_no_label = granular_ball[:, 1:]
    center = data_no_label.mean(0)  # 平均中心
    # 将半径r定义为平均距离而不是最大或最小距离的主要原因是，颗粒球的大小不容易受到离群样本的影响。
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))  # 所有数据距离平均中心的平均距离
    return center, radius


# 绘制粒球
def plot_gb(granular_ball_list, plt_type=0):
    color = {0: 'r', 1: 'k'}
    plt.figure(figsize=(5, 4))
    plt.axis([-1.2, 1.2, -1, 1])
    ball_num_str = str(len(granular_ball_list))
    for granular_ball in granular_ball_list:
        label, p = get_label_and_purity(granular_ball)
        center, radius = calculate_center_and_radius(granular_ball)

        if plt_type == 0:
            data0 = granular_ball[granular_ball[:, 0] == 0]
            data1 = granular_ball[granular_ball[:, 0] == 1]
            plt.plot(data0[:, 1], data0[:, 2], '.', color=color[0], markersize=5)
            plt.plot(data1[:, 1], data1[:, 2], '.', color=color[1], markersize=5)

        if plt_type == 0 or plt_type == 1:
            theta = numpy.arange(0, 2 * numpy.pi, 0.01)
            x = center[0] + radius * numpy.cos(theta)
            y = center[1] + radius * numpy.sin(theta)
            plt.plot(x, y, color[label], linewidth=0.8)

        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color=color[label])

    plt.show()


def evaluate_classification(granular_ball_list, test_data):
    """
    评估粒球分裂后的分类效果精度。

    参数:
    granular_ball_list: 分裂后的粒球列表，每个粒球是一个包含数据点的 numpy 数组。
    test_data: 测试数据集，格式为 numpy 数组，其中第一列是标签，其余列是特征。

    返回:
    accuracy: 分类精度。
    """
    # 提取特征和标签
    X = test_data[:, 1:]
    y = test_data[:, 0]

    # 将粒球中的数据点合并为一个训练集
    X_train = []
    y_train = []
    for ball in granular_ball_list:
        X_train.extend(ball[:, 1:])
        y_train.extend(ball[:, 0])
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # 使用 K-最近邻分类器
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 预测测试集
    y_pred = knn.predict(X_test)

    # 计算精度
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    data_mat = scipy.io.loadmat(r'dataset16.mat')
    keys = ['fourclass', 'svmguide1', 'diabetes', 'breastcancer', 'creditApproval',
            'votes', 'svmguide3', 'sonar', 'splice', 'mushrooms']

    # keys = ['fourclass', 'electrical', 'letter']
    for k in keys:  # 数据集
        print(k)
        times = 0
        num_gb = 0
        acc = []
        for i in range(0, 10):  # 重复10次取平均精度
            # 记录开始时间
            start = time.time()
            data = data_mat[k]
            # df = pd.read_csv(r"UCI\\" + k + ".csv", header=None)  # 加载数据集
            # data = df.values
            data[data[:, 0] == -1, 0] = 0
            # print(data)
            # print(numpy.unique(data[:, 0], axis=0))
            # 划分训练集、测试集
            train, test = train_test_split(data, test_size=0.2)
            # 数组去重
            train = numpy.unique(train, axis=0)
            # 纯度阈值
            # purity = 0.85
            purity = 0.95

            # 所有数据作为一个球输入gb_list
            granular_ball_list = [train]

            # 直接绘制输入数据
            # plot_gb(granular_ball_list)

            while True:
                # 记录一次划分前的粒球数
                ball_number_1 = len(granular_ball_list)

                granular_ball_list = splits(granular_ball_list, purity=purity, splitting_method='k-means')

                # 记录一次划分后的粒球数
                ball_number_2 = len(granular_ball_list)
                # 绘制每次划分的结果
                # plot_gb(granular_ball_list)
                if ball_number_1 == ball_number_2:
                    break

            # 绘制粒球数据
            # plot_gb(granular_ball_list)
            # 绘制粒球中心
            # plot_gb(granular_ball_list, 2)
            count_num = evaluate_classification(granular_ball_list, test)  # 测试精度
            # avg, std = mean_std(count_num)
            avg = np.array(count_num).mean()
            # print(avg)
            acc.append(avg)
            # print('acc:', acc)
            # 记录结束时间
            end = time.time()
            times = (end - start) + times
            num_gb = num_gb + len(granular_ball_list)
            # print('粒球数量：', len(granular_ball_list))
            # print('耗费时间', round((end - start) * 1000, 0))
            # break
        print('粒球数量：', round(num_gb / 10, 0))
        print('总平均耗时：%s' % (round(times / 10 * 1000, 0)))
        max_acc = np.max(acc)  # 最大精度
        mean_acc = np.mean(acc)  # 平均精度
        print("最终：", round(max_acc, 3), round(mean_acc, 3))


if __name__ == '__main__':
    # print(1)
    main()
