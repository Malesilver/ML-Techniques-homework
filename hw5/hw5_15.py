import numpy as np
import sys
from sklearn import svm
import matplotlib.pyplot as plt

def get_data(file):
    data = []
    sign = []
    with open(file, 'r') as f:
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

def convert_label(label,chosen_digit):
    size = label.shape[0]
    new_label = np.ones_like(label)
    for i in range(size):
        if label[i] != chosen_digit:
            new_label[i] = -1
    return new_label


if __name__ == "__main__":
    file_train = 'features.train'
    data_x,data_y = get_data(file_train)

    chosen_digit = 0
    data_y = convert_label(data_y,chosen_digit)

    lgc_list = [-6, -4, -2, 0, 2]
    w_length_list = []
    for i in lgc_list:
        clf = svm.LinearSVC(loss='hinge', C=10**i)
        clf.fit(data_x, data_y)
        w_length = np.sqrt(np.sum(clf.coef_**2))
        w_length_list.append(w_length)

    plt.figure(figsize=(10, 6))
    plt.plot(lgc_list, w_length_list, 'b')
    plt.plot(lgc_list, w_length_list, 'ro')
    for (c, w) in zip(lgc_list, w_length_list):
        plt.text(c + 0.1, w, str(round(w, 4)))
    plt.xlabel("log10(C)")
    plt.ylabel("||w||")
    plt.xlim(-8, 4)
    plt.title("||w|| versus log10(C)")
    plt.savefig("15.png")


