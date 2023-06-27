import numpy as np
import operator
from os import listdir


# 传参 in_x为测试数据，data_set为已有数据集，labels为标签向量，k为最近邻数目
def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]  # 获取数据集行数
    # 把in_x复制对应行数
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    # 计算每一行向量和
    sq_distance = sq_diff_mat.sum(axis=1)
    distances = sq_distance ** 0.5
    # 返回由小到大排序的下标
    sorted_dist_in_dicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_in_dicies[i]]
        # 获取字典内不同键对应键值，对对应键值加1
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 按第二个元素即第一个键值进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 读取文件至矩阵
def file_to_matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()  # 把行作为元素存为列表
    number_of_lines = len(array_of_lines)
    return_mat = np.zeros((number_of_lines, 3))  # 创建一n行3列列表
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()  # 去掉首尾空格
        list_from_line = line.split('\t')  # 按制表符分开
        return_mat[index, :] = list_from_line[0:3]  # 第index行等于list_from_line的数据
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


# 归一化处理
def auto_norm(data_set):
    min_vals = data_set.min(0)  # 返回每一列最小值组成的一维数组
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


# training set
def dating_class_test():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: ", classifier_result, "the real answer is: ", dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: ", error_count / float(num_test_vecs))
    print(error_count)


# convert 32*32 image to 1*1024 vector
def img_to_vector(filename):
    return_vector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector


# 手写字识别
def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('digits/trainingDigits')  # load the training set
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img_to_vector('digits/trainingDigits/' + file_name_str)
    test_file_list = listdir('digits/testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_set = img_to_vector('digits/testDigits/' + file_name_str)
        classifier_result = classify0(vector_under_set, training_mat, hw_labels, 3)
        if classifier_result != class_num_str:
            error_count += 1
    print("\nthe total number of errors is: ", error_count)
    print("\nthe total error rate is: ", error_count / float(m_test))


group, labels = create_data_set()
a = classify0([0, 0], group, labels, 3)
dating_data_mat, dating_labels = file_to_matrix('datingTestSet2.txt')
dating_class_test()
handwriting_class_test()
