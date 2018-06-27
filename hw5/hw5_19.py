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
    file_test = 'features.test'
    data_x, data_y = get_data(file_train)
    test_x, test_y = get_data(file_test)

    chosen_digit = 0
    data_y = convert_label(data_y, chosen_digit)
    test_y = convert_label(test_y, chosen_digit)

    lg_gamma_list = [0, 1, 2, 3, 4]
    err_test_list = np.zeros(len(lg_gamma_list))
    for i in range(len(lg_gamma_list)):
        lg_gamma = lg_gamma_list[i]
        clf = svm.SVC(C=0.1, kernel='rbf', gamma=10**lg_gamma)
        clf.fit(data_x, data_y)
        y_predict = clf.predict(test_x)         #求错误率也可用clf.score()
        err_test = get_err(y_predict, test_y)
        err_test_list[i] = err_test

    plt.figure(figsize=(10, 6))
    plt.plot(lg_gamma_list, err_test_list, 'b')
    plt.plot(lg_gamma_list, err_test_list, 'ro')
    for (c, w) in zip(lg_gamma_list, err_test_list):
        plt.text(c + 0.1, w, str(round(w, 4)))
    plt.xlabel("log10(gamma)")
    plt.ylabel("Eout")
    plt.xlim(-1, 5)
    plt.ylim(ymax=0.19)
    plt.title("Eout versus log10(C)")
    plt.savefig("19.png")