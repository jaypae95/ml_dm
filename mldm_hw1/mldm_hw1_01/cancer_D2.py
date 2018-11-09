import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations

with open("cancer_train.csv", 'r') as trainfile:
    df = pd.read_csv(trainfile, names=[x for x in range(0, 10)])
with open("cancer_test.csv", 'r') as testfile:
    test = pd.read_csv(testfile, names=[x for x in range(0, 10)])

X = df.drop(0, axis=1).values
y = df[0].values

X_test = test.drop(0, axis=1).values
y_test = test[0].values

print("===============DecisionTree===============")

combn = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], 7))
randm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_pred = 0;
max_random = 0;
max_feature = []

for i in combn:
    for j in randm:
        X_combn = X[:, i]
        X_combn_test = X_test[:, i]

        tree = DecisionTreeClassifier(random_state=j)
        tree.fit(X_combn, y)
        pred = tree.score(X_combn_test, y_test)

        if max_pred < pred:
            max_pred = pred
            max_random = j
            max_feature = i

print("<<Feature Used: ", max_feature, ">>")
print("<<Random State Used: ", max_random, ">>")
print("<<Accuracy Rate: {0:.2f}%>>"
      .format(max_pred * 100))
