import random

import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_array = line.strip().split()
        data_mat.append([1.0, float(line_array[0]), float(line_array[1])])
        label_mat.append(int(line_array[2]))
    return data_mat, label_mat


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 定义梯度上升函数
def grad_ascent(data_mat_in, class_labels):
    data_mat = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_mat)
    alpha = 0.001  # 步长
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = label_mat - h  # 误差
        weights = weights + alpha * data_mat.transpose() * error
    weights = np.array(weights)
    return weights


def plot_best_fit(weights):
    data_mat, label_mat = load_data_set()
    data_array = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_array[i, 1])
            y_cord1.append(data_array[i, 2])
        else:
            x_cord2.append(data_array[i, 1])
            y_cord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    weights_ls = weights.tolist()
    y = (-np.ones(60) * weights_ls[0] - weights_ls[1] * x) / weights_ls[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def stochastic_grad_ascent_0(data_mat, class_labels):
    data_mat = np.array(data_mat)
    class_labels = np.array(class_labels)
    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for j in range(50):  # 次数过少时不收敛
        for i in range(100):
            h = sigmoid(sum(data_mat[i] * weights))
            error = class_labels[i] - h
            weights = weights + alpha * error * data_mat[i]
    return weights


def stochastic_grad_ascent_1(data_mat, class_labels, num_iter=150):
    data_mat = np.array(data_mat)
    class_labels = np.array(class_labels)
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.001
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * data_mat[rand_index] * error
            del (data_index[rand_index])
    return weights


data_mat, label_mat = load_data_set()
weights = stochastic_grad_ascent_1(data_mat, label_mat)
plot_best_fit(weights)
