import numpy as np
import sys
import numpy.random as rd
import matplotlib.pyplot as plt


def read_data(file_name):
    data = []
    sign = []
    with open(file_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            for i in range(len(items)):
                items[i] = float(items[i])
            data.append(items[0:-1])
            sign.append(items[-1])
    data = np.array(data)
    sign = np.array(sign)
    if data.shape[0] != sign.shape[0]:
        sys.exit(-1)
    return data, sign


def get_sign(d):
    # 符号函数，大于0返回+1，否则返回-1
    # d: 一维数组
    sign = np.ones_like(d)
    for i, val in enumerate(d):
        if val <= 0:
            sign[i] = -1
    return sign


def get_err(sign_prim, sign_pred):
    if sign_prim.shape[0] != sign_pred.shape[0]:
        sys.exit(-1)
    size = sign_prim.shape[0]
    res = np.multiply(sign_pred, sign_prim)
    err_count = (size - np.sum(res))/2
    return err_count/size


def knbor(k, train_x, train_y, data_x):
    if train_x.shape[1] != data_x.shape[1]:
        sys.exit(-1)
    y_pred = np.zeros(data_x.shape[0])
    for i in range(data_x.shape[0]):
        dist = np.zeros(train_x.shape[0])
        for j in range(train_x.shape[0]):
            square = (data_x[i] - train_x[j]) ** 2
            dist[j] = np.sum(square)
        knn_id = np.argsort(dist)[:k]
        knn_sign = train_y[knn_id]
        y_pred[i] = 1 if np.sum(knn_sign) > 0 else -1
    return y_pred


if __name__ == '__main__':
    file_train = 'knn_train.dat'
    file_test = 'knn_test.dat'
    train_x, train_y = read_data(file_train)
    test_x, test_y = read_data(file_test)

    k = np.array([1, 3, 5, 7, 9])
    E_in = np.zeros(k.shape[0])
    E_out = np.zeros(k.shape[0])
    for i, val in enumerate(k):
        y_knn_train = knbor(k=val, train_x=train_x, train_y=train_y, data_x=train_x)
        error_train = get_err(train_y, y_knn_train)
        E_in[i] = error_train
        y_knn_test = knbor(k=val, train_x=train_x, train_y=train_y, data_x=test_x)
        error_test = get_err(test_y, y_knn_test)
        E_out[i] = error_test
    print(E_in)
    print(E_out)
'''
[ 0.    0.1   0.16  0.15  0.14]
[ 0.344  0.299  0.316  0.322  0.303]
'''