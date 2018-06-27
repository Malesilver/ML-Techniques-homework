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


def data_split(data_x,data_y,num=1000):
    size = data_x.shape[0]
    arr = np.arange(0,size)
    np.random.shuffle(arr)
    val_index = arr[0:num]

    val_x = []
    val_y = []
    train_x = []
    train_y = []
    for i in range(size):
        if i in val_index:
            val_x.append(data_x[i])
            val_y.append(data_y[i])
        else:
            train_x.append(data_x[i])
            train_y.append(data_y[i])
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y, val_x, val_y


if __name__ == "__main__":
    file_train = 'features.train'
    file_test = 'features.test'
    data_x, data_y = get_data(file_train)
    test_x, test_y = get_data(file_test)

    chosen_digit = 0
    data_y = convert_label(data_y, chosen_digit)
    test_y = convert_label(test_y, chosen_digit)

    lg_gamma_list = [0, 1, 2, 3, 4]
    chosen_gamma = []
    for i in range(100):
        err_val_list = np.zeros(len(lg_gamma_list))
        train_x, train_y, val_x, val_y = data_split(data_x, data_y, num=1000)
        for j in range(len(lg_gamma_list)):
            lg_gamma = lg_gamma_list[j]
            clf = svm.SVC(C=0.1, kernel='rbf', gamma=10**lg_gamma)
            clf.fit(train_x, train_y)
            right_score = clf.score(val_x,val_y)
            err_val_list[j] = 1-right_score
        index = np.argmin(err_val_list)
        chosen_gamma.append(np.array(lg_gamma_list)[index])

    times = []
    for i in lg_gamma_list:
        times.append(chosen_gamma.count(i))
    plt.figure(figsize=(10, 6))
    plt.bar(left=lg_gamma_list, height=times, width=1, align="center", yerr=0.000001)
    for (c, w) in zip(lg_gamma_list, times):
        plt.text(c, w * 1.03, str(round(w, 4)))
    plt.xlabel("log10(gamma)")
    plt.ylabel("the number of chosen times")
    plt.xlim(-1, 5)
    plt.ylim(0, 80)
    plt.title("the number of chosen times for gamma")
    plt.savefig("20.png")
