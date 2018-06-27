import numpy as np
import numpy.random as rd
import sys
import matplotlib.pyplot as plt


def read_data(file_name):
    data = []
    sign = []
    with open(file_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            for i in range(len(items)):
                items[i] = float(items[i])
            items.insert(0, 1)              # x_0 = 1
            data.append(items[0:-1])
            sign.append(items[-1])
    data = np.array(data)
    sign = np.array(sign)
    if data.shape[0] != sign.shape[0]:
        sys.exit(-1)
    return data, sign


def der_tanh(x):
    tanh_x = (np.exp(x)-np.exp(-1*x)) / (np.exp(x)+np.exp(-1*x))
    res = 1 - tanh_x**2
    return res


def forward(x_list, w_list, s_list, layer, dim):
    for l in range(1, layer):
        x = np.ones(dim[l] + 1)
        s = np.zeros(dim[l] + 1)
        for j in range(1, dim[l]+1):
            s[j] = np.dot(x_list[l-1], w_list[l][:, j-1])
            x[j] = (np.exp(s[j])-np.exp(-1*s[j])) / (np.exp(s[j])+np.exp(-1*s[j]))
        x_list.append(x)
        s_list.append(s)


def backward(delta_list, w_list, s_list, x_list, y, layer, dim):
    for l in range(1, layer):
        delta = np.zeros(dim[l]+1)
        delta_list.append(delta)
    # calculate the delta of the last layer
    for k in range(1, dim[layer-1]+1):
        delta_list[layer-1][k] = (-2) * (y - x_list[layer-1][k]) * der_tanh(s_list[layer-1][k])
    # calculate delta backward
    l = layer-2
    while l >= 1:
        for j in range(1, dim[l]+1):
            delta_list[l][j] = np.dot(delta_list[l+1][1:], w_list[l+1][j]) * der_tanh(s_list[l][j])
        l = l - 1


def cal_gd(gd_list, x_list, delta_list, layer):
    for l in range(1, layer):
        gd = np.array(np.dot(np.matrix(x_list[l-1]).transpose(), np.matrix(delta_list[l][1:])))
        gd_list.append(gd)


def nnet_bp(data_x, data_y, layer, dim, eta, r, T=5000):
    dim = np.array([int(i) for i in dim])
    w_list = [np.array([])]
    # initialize w_list
    for l in range(1, layer):
        if r == 0:
            w_list.append(np.zeros((dim[l-1]+1, dim[l])))
        else:
            w_list.append(rd.uniform(-1*r, r, size=(dim[l-1]+1, dim[l])))
    # iterative
    for t in range(T):
        x_list = []
        n = rd.randint(0, data_x.shape[0])
        x_list.append(data_x[n])
        s_list, delta_list, gd_list = [np.array([])], [np.array([])], [np.array([])]
        forward(x_list, w_list, s_list, layer, dim)
        backward(delta_list, w_list, s_list, x_list, data_y[n], layer, dim)
        cal_gd(gd_list, x_list, delta_list, layer)
        # update w_list
        for l in range(1, layer):
            w_list[l] = w_list[l] - eta * gd_list[l]
    # return nnet
    return w_list


def nnet_predict(data_x, w_list, layer, dim):
    y_pred = np.zeros(data_x.shape[0])
    for i in range(data_x.shape[0]):
        x_list = list([])
        x_list.append(data_x[i])
        s_list = [np.array([])]
        forward(x_list=x_list, w_list=w_list, s_list=s_list, layer=layer, dim=dim)
        y_pred[i] = x_list[-1][1]
    return y_pred


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


if __name__ == "__main__":
    file_train = 'nnet_train.dat'
    file_test = 'nnet_test.dat'
    train_x, train_y = read_data(file_train)
    test_X, test_y = read_data(file_test)

    eta = np.array([0.001, 0.01, 0.1, 1, 10])
    r = 0.1
    M = 3
    dim = np.array([train_x.shape[1]-1, M, 1])
    layer_n = 3

    # err_avg = np.zeros_like(M)            # don't use zeros_like
    err_avg = np.zeros(eta.shape[0])
    for i, val in enumerate(eta):
        times = 20
        err_test = np.zeros(times)
        for n in range(times):
            w_arr = nnet_bp(train_x, train_y, layer=layer_n, dim=dim, eta=val, r=r, T=50000)
            y_predict = nnet_predict(data_x=test_X, w_list=w_arr, layer=layer_n, dim=dim)
            error = get_err(test_y, get_sign(y_predict))
            err_test[n] = error
        err_avg[i] = np.sum(err_test)/times
        print(val)
    print(err_avg)
    min_id = np.argmin(err_avg)
    print(eta[min_id])
'''
[ 0.136   0.036   0.0378  0.483   0.4972]
0.01
'''
