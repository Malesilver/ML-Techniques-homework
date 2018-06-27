import numpy as np
import sys
from sklearn import svm
import matplotlib.pyplot as plt


def get_data(file):
    data = []
    sign = []
    with open(file,'r') as f:
        for line in f:
            words = line.strip().split()
            for i in range(len(words)):
                words[i] = float(words[i])
            data.append(words[1:])
            sign.append(words[0])
    data = np.array(data)
    sign = np.array(sign)
    if data.shape[0] != sign.shape[0]:
        sys.exit(-1)
    return data, sign


def convert_label(label, chosen_digit):
    size = label.shape[0]
    new_label = np.ones_like(label)
    for i in range(size):
        if label[i] != chosen_digit:
            new_label[i] = -1
    return new_label


def get_err(y_predict, y_prim):
    if y_prim.shape[0] != y_predict.shape[0]:
        sys.exit(-1)

    size = y_prim.shape[0]
    temp = np.multiply(y_predict, y_prim)
    err_count = (size - np.sum(temp))/2
    return err_count/size


if __name__ == "__main__":
    file_train = 'features.train'
    data_x, data_y = get_data(file_train)

    chosen_digit = 8
    data_y = convert_label(data_y,chosen_digit)

    lgc_list = [-6, -4, -2, 0, 2]
    err_train_list = np.zeros(len(lgc_list))
    alpha_sum_list = []
    for i in range(len(lgc_list)):
        lgc = lgc_list[i]
        clf = svm.SVC(C=10**lgc, kernel='poly', degree=2, gamma=1, coef0=0)
        clf.fit(data_x, data_y)
        # E_in
        y_predict = clf.predict(data_x)
        err_train = get_err(y_predict, data_y)
        err_train_list[i] = err_train
        # sum of alpha
        # alpha = np.multiply(clf.dual_coef_[0], data_y[clf.support_])
        alpha = np.sum(np.abs(clf.dual_coef_))
        alpha_sum = np.sum(alpha)
        alpha_sum_list.append(alpha_sum)

    #16
    plt.figure(figsize=(10, 6))
    plt.plot(lgc_list, err_train_list, 'b')
    plt.plot(lgc_list, err_train_list, 'ro')
    for (c, e) in zip(lgc_list, err_train_list):
        plt.text(c + 0.1, e, str(round(e, 4)))
    plt.xlabel("log10(C)")
    plt.ylabel("Ein")
    plt.xlim(-8, 4)
    plt.title("Ein versus log10(C)")
    plt.savefig("16.png")

    #17
    plt.figure(figsize=(10, 6))
    plt.plot(lgc_list, alpha_sum_list, 'b')
    plt.plot(lgc_list, alpha_sum_list, 'ro')
    for (c, a) in zip(lgc_list, alpha_sum_list):
        plt.text(c + 0.1, a, str(round(a, 6)))
    plt.xlabel("log10(C)")
    plt.ylabel("sum of alpha")
    plt.xlim(-8, 4)
    plt.title("sum of alpha versus log10(C)")
    plt.savefig("17.png")