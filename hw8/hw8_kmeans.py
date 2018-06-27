import numpy as np
import sys
import numpy.random as rd
import matplotlib.pyplot as plt


def read_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            for i in range(len(items)):
                items[i] = float(items[i])
            data.append(items)
    data = np.array(data)
    return data


def kmeans(k, data):
    prototype_id = rd.permutation(np.arange(0, data.shape[0]))[:k]
    prototype = data[prototype_id]                  # initialize prototypes
    subset = [list([]) for i in range(k)]           # initialize subsets
    subset_reserve = [list([]) for i in range(k)]

    while True:
        # update subset
        for i in range(data.shape[0]):
            dist = np.zeros(k)              # distance to each prototype
            for j in range(k):
                dist[j] = np.sum((data[i] - prototype[j])**2)
            min_id, min = 0, dist[0]
            for n, val in enumerate(dist):
                if val < min:
                    min = val
                    min_id = n
            subset[min_id].append(i)        # add in the subset related to the closest prototype
        # check subset
        s = set()
        for n in range(k):
            s = s | set(subset[n])
        if len(s) != data.shape[0]:
            sys.exit('subset error')
        # end is True: no change of subset
        end = False
        for n in range(k):
            if len(set(subset[n]) ^ set(subset_reserve[n])) == 0:
                end = True
            else:
                end = False
                break
        # end is False: update prototype; end is True: stop
        if end is False:
            subset_reserve = subset
            # update prototype
            for n in range(k):
                prototype[n] = np.sum(data[np.array(subset[n])], axis=0) / len(subset[n])
            # initialize subset
            subset = [list([]) for i in range(k)]
        else:
            break
    return prototype, subset


def get_error(k, prototype, subset, data):
    total = 0
    for i in range(k):
        dist = np.zeros(len(subset[i]))
        for j in range(len(subset[i])):
            dist[j] = np.sum((data[subset[i][j]] - prototype[i])**2)
        total = total + np.sum(dist)
    return total/data.shape[0]


if __name__ == '__main__':
    file = 'knn_train.dat'
    data = read_data(file)

    times = 500
    k = np.array([2, 4, 6, 8, 10])
    average_E_in = np.zeros(k.shape[0])
    for i, val in enumerate(k):
        E_in = np.zeros(times)
        for t in range(times):
            prototype, subset = kmeans(k=val, data=data)
            E_in[t] = get_error(k=val, prototype=prototype, subset=subset, data=data)
        average_E_in[i] = np.sum(E_in)/times
    print(average_E_in)
'''
[ 2.81612282  2.46065905  2.20810297  2.01435276  1.85516804]
'''




