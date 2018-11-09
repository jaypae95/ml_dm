import numpy as np
from operator import eq
import collections
import sys


def train(k=5, distance='euclidean', cv=5):
    def fit_(X, y):
        if cv <= 1:  # no cv
            return [X, y, k, distance]
        else:  # if k-fold cv
            accuracy = []
            print("="*13, "Cross validation Started", "="*13)
            for split in range(1, cv+1):
                index = int((len(X)/cv)*split)-1  # split data index
                pre_index = int(index - len(X)/cv)+1  # split data pre-index

                if split == cv:
                    # last data part
                    # if data length is 801, the last one would not be computed.
                    # this job is to prevent that case.
                    index = int(len(X))

                # data split
                delete_index = [i for i in range(pre_index, index + 1)]
                X_test_cv = X[pre_index:index+1]
                y_test_cv = y[pre_index:index+1]
                X_cv = np.delete(X, delete_index, axis=0)
                y_cv = np.delete(y, delete_index, axis=0)
                train = [X_cv, y_cv, k, distance]
                print("\n", pre_index, index)

                # test
                ac, pr = predict(X_test_cv, y_test_cv, train)
                accuracy.append(ac)
            # average accuracy
            mean_accuracy = round(float(np.mean(accuracy)*100), 2)
            print("<<Cross validation average accuracy = ", mean_accuracy, "% >>")
            return [X, y, k, distance]

    return fit_


def get_distance(X_test, train, i):
    dist = []
    if eq(train[3], 'euclidean'):  # euclidean(p=2)
        for j in range(len(train[0])):
            dist.append([np.sqrt(sum((X_test[i, :] - train[0][j, :]) ** 2)), j])
    elif eq(train[3], 'manhattan'):  # manhattan(p=1)
        for j in range(len(train[0])):
            dist.append([sum(abs(X_test[i, :] - train[0][j, :])), j])
    elif eq(train[3], 'l_infinity'):  # L_infinity(p=infinity)
        for j in range(len(train[0])):
            dist.append([max(abs(X_test[i, :] - train[0][j, :])), j])
    else:  # distance not defined
        print("distance type undefined.\n Exit program...\n")
        sys.exit(0)
    return dist


def predict(X_test, y_test, train):
    maj_result = []
    star_count = 0
    for i in range(len(X_test)):
        dist = get_distance(X_test, train, i)
        dist = sorted(dist)
        real = []
        for m in range(train[2]):
            real.append(train[1][dist[m][1]])

        if i % 50 == 0:  # just to show the program is in progress
            print("*" * star_count)
            star_count += 1

        # to pick majority value
        c = collections.Counter(sorted(real))
        maj = c.most_common(1)[0][0]
        maj_result.append(maj)

    count = 0
    n = 0
    for n in range(len(maj_result)):  # to compare real value and predict value
        if maj_result[n] == y_test[n]:
            count += 1

    return [count/(n+1), maj_result]  # accuracy
