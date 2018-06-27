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


def rbf_uniform(gamma, train_x, train_y, data_x):
    if train_x.shape[1] != data_x.shape[1]:
        sys.exit(-1)
    y_pred = np.zeros(data_x.shape[0])
    for i in range(data_x.shape[0]):
        res_Gaussian = np.zeros(train_x.shape[0])
        for j in range(train_x.shape[0]):
            dist = np.sum((data_x[i] - train_x[j])**2)
            res_Gaussian[j] = np.exp(-1*gamma*dist)
        y_pred[i] = 1 if np.dot(res_Gaussian, train_y) > 0 else -1
    return y_pred


if __name__ == '__main__':
    file_train = 'knn_train.dat'
    file_test = 'knn_test.dat'
    train_x, train_y = read_data(file_train)
    test_x, test_y = read_data(file_test)

    gamma = np.array([0.001, 0.1, 1, 10, 100])
    E_in = np.zeros(gamma.shape[0])
    E_out = np.zeros(gamma.shape[0])
    for i, val in enumerate(gamma):
        y_rbf_train = rbf_uniform(gamma=val, train_x=train_x, train_y=train_y, data_x=train_x)
        E_in[i] = get_err(train_y, y_rbf_train)
        y_rbf_test = rbf_uniform(gamma=val, train_x=train_x, train_y=train_y, data_x=test_x)
        E_out[i] = get_err(test_y, y_rbf_test)
    print(E_in)
    print(E_out)
'''
[ 0.45  0.45  0.02  0.    0.  ]
[ 0.467  0.448  0.288  0.346  0.344]
'''