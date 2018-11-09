import pandas as pd
from sklearn.tree import DecisionTreeClassifier

with open("cancer_train.csv", 'r') as trainfile:
    df = pd.read_csv(trainfile, names=[x for x in range(0, 10)])
with open("cancer_test.csv", 'r') as testfile:
    test = pd.read_csv(testfile, names=[x for x in range(0, 10)])

X = df.drop(0, axis=1).values
y = df[0].values

X_test = test.drop(0, axis=1).values
y_test = test[0].values

print("===============DecisionTree===============")

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X, y)
pred = tree.score(X, y)

print("<<Train Accuracy Rate: {0:.2f}%>>".format(pred*100))

pred = tree.score(X_test, y_test)

print("<<Test Accuracy Rate: {0:.2f}%>>".format(pred*100))
