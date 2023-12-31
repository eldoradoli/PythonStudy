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


# def sigmoid(x):
#     return 1.0 / (1 + np.exp(-x))
# 优化后的激活函数,避免了出现极大的数据溢出
def sigmoid(inx):
    if inx >= 0:
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))


# 梯度上升函数
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


def classify_vec(inx, weights):
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def horse_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):  # number of features
            line_arr.append(float(current_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(current_line[21]))
    training_weights = stochastic_grad_ascent_1(np.array(training_set), training_labels, 1000)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))
        if int(classify_vec(np.array(line_arr), training_weights)) != int(current_line[21]):
            error_count += 1
    error_rate = float(error_count / num_test_vec)
    print("the error rate of this test is: ", error_rate)
    return error_rate


data_mat, label_mat = load_data_set()
weights = stochastic_grad_ascent_1(data_mat, label_mat)
plot_best_fit(weights)
horse_test()
