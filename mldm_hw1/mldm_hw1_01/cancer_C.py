import pandas as pd
from sklearn.svm import SVC
with open("cancer_train.csv", 'r') as trainfile:
    df = pd.read_csv(trainfile, names=[x for x in range(0, 10)])
with open("cancer_test.csv", 'r') as testfile:
    test = pd.read_csv(testfile, names=[x for x in range(0, 10)])

X = df.drop(0, axis=1).values
y = df[0].values

X_test = test.drop(0, axis=1).values
y_test = test[0].values

print("===============SVM===============")
clf = SVC(kernel='linear')
clf.fit(X, y)
pred = clf.score(X, y)
print("<<Train Accuracy Rate: {0:.2f}%>>".format(pred*100))

pred = clf.score(X_test, y_test)
print("<<Test Accuracy Rate: {0:.2f}%>>".format(pred*100))
