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


def get_w_length(data_x, data_y, alpha, sv_id, gamma):
    Q = np.zeros((sv_id.shape[0],sv_id.shape[0]))
    for i in range(sv_id.shape[0]):
        n = i
        for j in range(sv_id.shape[0]):
            m = j
            v = data_x[n]-data_x[m]
            v_len = np.sum(v**2)
            e = -1*gamma*v_len
            Q_ij = data_y[n]*data_y[m]*np.exp(e)
            Q[i, j] = Q_ij
    wTW = np.dot(np.dot(Q,alpha),alpha)
    w_length = np.sqrt(wTW)
    return w_length


if __name__ == "__main__":
    file_train = 'features.train'
    data_x, data_y = get_data(file_train)

    chosen_digit = 0
    data_y = convert_label(data_y,chosen_digit)

    lgc_list = [-3, -2, -1, 0, 1]
    dis_list = np.zeros(len(lgc_list))
    wTw_list = np.zeros(len(lgc_list))
    ## objective value from libsvm
    obj_list = np.array([-2.356303, -20.712043, -150.327955, -1375.660852, -13460.946158])
    vio_sum_list = np.zeros(len(lgc_list))

    for i in range(len(lgc_list)):
        lgc = lgc_list[i]
        clf = svm.SVC(C=10**lgc, kernel='rbf',  gamma=100)
        clf.fit(data_x, data_y)

        alpha = np.abs(clf.dual_coef_.reshape((clf.dual_coef_.shape[1],)))
        dec_val = clf.decision_function(clf.support_vectors_)
        w_len = get_w_length(data_x, data_y, alpha, sv_id=clf.support_, gamma=100)
        dis = np.abs(dec_val)/w_len
        # dis = np.abs(dec_val)
        for j in range(alpha.shape[0]):
            if 0 < alpha[j] < 10**lgc:
                dis_list[i]=dis[j]
                break

        vio_sum = (obj_list[i] - 0.5*w_len*w_len)/(10**lgc)
        vio_sum_list[i] = vio_sum


    # print(dis_list)
    # print(vio_sum_list)
    '''
[  3.39000461e+00   3.37658516e-01   3.89140148e-02   5.39940954e-03
   5.82125776e-04]
[  -2399.82113183   -2509.62187242   -4804.65958988  -18532.82556137
 -148817.99248291]
    '''

    plt.figure(figsize=(10, 6))
    plt.plot(lgc_list, dis_list, 'b')
    plt.plot(lgc_list, dis_list, 'ro')
    for (c, w) in zip(lgc_list, dis_list):
        plt.text(c + 0.1, w, str(round(w, 4)))
    plt.xlabel("log10(C)")
    plt.ylabel("free sv's function distance to hyperplane")
    plt.xlim(-5, 3)
    plt.ylim(ymin=-1, ymax=4)
    plt.title("free sv's function distance to hyperplane versus log10(C)")
    plt.savefig("18.png")