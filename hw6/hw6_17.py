import numpy as np
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
            data.append(items[0:-1])
            sign.append(items[-1])
    data = np.array(data)
    sign = np.array(sign)
    if data.shape[0] != sign.shape[0]:
        sys.exit(-1)
    return data, sign


def get_sign(d):                    #符号函数，大于0返回+1，否则返回-1
    sign = np.ones(d.shape)
    for i, val in enumerate(d):
        if val <= 0:
            sign[i] = -1
    return sign


def get_theta(data):
    theta_num = data.shape[0]
    theta = np.zeros(theta_num)
    data_lh = sorted(data)
    for i in range(theta_num-1):
        theta[i] = (data_lh[i] + data_lh[i+1])*0.5
    theta[-1] = data_lh[-1]+(data_lh[-1]-theta[-2])
    return theta


def get_err(sign_prim, sign_pred):
    if sign_prim.shape[0] != sign_pred.shape[0]:
        sys.exit(-1)
    size = sign_prim.shape[0]
    res = np.multiply(sign_pred,sign_prim)
    err_count = (size - np.sum(res))/2
    return err_count/size


def get_err_weight(weight, data, sign, theta, s):
    size = data.shape[0]
    sign_pred = s * get_sign(data - theta)
    res = np.multiply(sign, sign_pred)
    err_01 = np.zeros_like(res)
    for i, val in enumerate(res):
        if val == -1:
            err_01[i] = 1
        elif val == 1:
            pass
        else:
            sys.exit(-1)
    err_sum = np.dot(err_01, weight)
    return err_sum/size


def dec_stump_1d(data, sign, weight):
    theta = get_theta(data)
    theta_num = theta.shape[0]
    err_array = np.zeros((2, theta_num))
    for i in range(theta_num):
        err_array[0][i] = get_err_weight(weight, data, sign, theta[i], s=1)
        err_array[1][i] = get_err_weight(weight, data, sign, theta[i], s=-1)

    min_0_id = np.argmin(err_array[0])
    min_1_id = np.argmin(err_array[1])
    min_0 = err_array[0][min_0_id]
    min_1 = err_array[1][min_1_id]

    if min_0 < min_1:
        err_min = min_0
        s_best = 1
        theta_best = theta[min_0_id]
    else:
        err_min = min_1
        s_best = -1
        theta_best = theta[min_1_id]

    return s_best, theta_best, err_min


def dec_stump(data_x, data_y, weight):
    n_feature = data_x.shape[1]
    s_arr = np.zeros(n_feature)
    theta_arr = np.zeros(n_feature)
    err_arr = np.zeros(n_feature)
    for i in range(n_feature):
        feature = data_x[:, i]
        sign = data_y
        s_arr[i], theta_arr[i], err_arr[i] = dec_stump_1d(feature, sign, weight)

    err_min_id = np.argmin(err_arr)
    err_min = err_arr[err_min_id]
    s_chosen = s_arr[err_min_id]
    theta_chosen = theta_arr[err_min_id]
    i_chosen = err_min_id

    return s_chosen, theta_chosen, i_chosen, err_min


def adaboost_stump(data_x, data_y, T=300):
    size = data_x.shape[0]

    alpha_list = np.zeros(T)
    s_list = np.zeros(T)
    theta_list = np.zeros(T)
    i_list = np.zeros(T)

    weight = np.ones(size) / size
    for k in range(T):
        s, theta, i, err = dec_stump(data_x, data_y, weight)
        epsilon = (err * size) / np.sum(weight)
        if epsilon == 0:
            print('epsilon = 0')
            sys.exit(-1)
        # updata weight
        sign_pred = s * get_sign(data_x[:, i] - theta)
        res = np.multiply(data_y, sign_pred)
        for j, val in enumerate(res):
            if val == -1:
                weight[j] = weight[j] * np.sqrt((1-epsilon)/epsilon)
            elif val == 1:
                weight[j] = weight[j] / np.sqrt((1-epsilon)/epsilon)
            else:
                sys.exit(-1)
        # get alpha
        alpha = np.log((1 - epsilon) / epsilon) * 0.5
        # save model
        alpha_list[k] = alpha
        s_list[k] = s
        theta_list[k] = theta
        i_list[k] = i

    return alpha_list, s_list, theta_list, i_list


def plot_line_chart(X=np.arange(0,300,1),Y=np.arange(0,300,1),
                    nameX="t",nameY="Ein(gt)",saveName="12.png"):
    X = list(X)
    Y = list(Y)
    plt.figure(figsize=(30, 12))
    plt.plot(X, Y, 'b')
    plt.plot(X, Y, 'ro')
    plt.xlim((X[0] - 1, X[-1] + 1))
    for (x, y) in zip(X, Y):
        if(x % 10 == 0): plt.text(x + 0.1, y, str(round(y, 4)))
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    plt.title(nameY + " versus " + nameX)
    plt.savefig(saveName)
    return


if __name__ == '__main__':
    file_train = 'adaboost_train.dat'
    file_test = 'adaboost_test.dat'
    data_x, data_y = read_data(file_train)
    test_x, test_y = read_data(file_test)
    T = 300
    alpha_list, s_list, theta_list, i_list = adaboost_stump(data_x, data_y, T=T)

    err_test_list = np.zeros(T)
    for n in range(T):
        s = s_list[n]
        theta = theta_list[n]
        i = int(i_list[n])
        sign_pred = s * get_sign(test_x[:, i] - theta)
        err_test = get_err(test_y, sign_pred)
        err_test_list[n] = err_test


    plot_line_chart(Y=err_test_list, nameY="Eout(gt)", saveName="17.png")
    print("Eout(g1):", err_test_list[0])






