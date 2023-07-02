import random

import numpy as np
import re


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


# 创建文档不重复词汇列表
def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 求集合并
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word', word, 'is not in the vocabulary')
    return return_vec


def train_naive_bayes(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive


def classify_naive_bayes(vec_to_classify, p0_vec, p1_vec, p_class_1):
    p1 = sum(vec_to_classify * p1_vec) + np.log(p_class_1)
    p0 = sum(vec_to_classify * p0_vec) + np.log(1.0 - p_class_1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_naive_bayes():
    list_post, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_post)
    train_mat = []
    for post_in_doc in list_post:
        train_mat.append(set_of_words_to_vec(my_vocab_list, post_in_doc))
    p0_vec, p1_vec, p_abuse = train_naive_bayes(np.array(train_mat), np.array(list_classes))
    test_entry_0 = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words_to_vec(my_vocab_list, test_entry_0))
    print(test_entry_0, 'classify as: ', classify_naive_bayes(this_doc, p0_vec, p1_vec, p_abuse))
    test_entry_1 = ['stupid', 'garbage']
    this_doc = np.array(set_of_words_to_vec(my_vocab_list, test_entry_1))
    print(test_entry_1, 'classify as: ', classify_naive_bayes(this_doc, p0_vec, p1_vec, p_abuse))


def bag_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


# 接受一字符串将其解析为token列表，并去除长度小于3的token
def text_parse(bit_string):
    # list_of_tokens = re.split(r'\W', bit_string)
    list_of_tokens = re.findall('[a-zA-Z]*', bit_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/' + str(i) + '.txt').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/' + str(i) + '.txt').read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    # 从50个文件中选10个作为测试集并从原集合中删去，剩余作为训练集
    training_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]
    train_mat = []
    train_class = []
    for doc_index in training_set:
        train_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0_vec, p1_vec, p_spam = train_naive_bayes(np.array(train_mat), np.array(train_class))
    error_count = 0
    for doc_index in test_set:
        word_vec = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_naive_bayes(np.array(word_vec), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            error_count += 1
            print('classification error', doc_list[doc_index])
    print('the error rate is: ', float(error_count) / len(test_set))


spam_test()
