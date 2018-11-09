import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

with open("cancer_train.csv", 'r') as trainfile:
    df = pd.read_csv(trainfile, names=[x for x in range(0, 10)])
with open("cancer_test.csv", 'r') as testfile:
    test = pd.read_csv(testfile, names=[x for x in range(0, 10)])

X = df.drop([0, 5, 9], axis=1).values
y = df[0].values
X_test = test.drop([0, 5, 9], axis=1).values
y_test = test[0].values

print("===============DecisionTree===============")

tree = DecisionTreeClassifier(random_state=10)
tree.fit(X, y)
pred = tree.score(X_test, y_test)

y_pred = tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\n============Confusion Matrix============")
print(cm)
print("\n=================REPORT=================")

precision_0 = cm[0, 0]/(cm[0, 0] + cm[1, 0])
recall_0 = cm[0, 0]/(cm[0, 0] + cm[0, 1])
f1score_0 = 2*(precision_0*recall_0)/(precision_0+recall_0)

precision_1 = cm[1, 1]/(cm[1, 1] + cm[0, 1])
recall_1 = cm[1, 1]/(cm[1, 1] + cm[1, 0])
f1score_1 = 2*(precision_1*recall_1)/(precision_1+recall_1)

report = """
precision = %.2f
recall = %.2f
f1score = %.2f
"""

print("Class 0 : ", report % (precision_0, recall_0, f1score_0))
print("Class 1 : ", report % (precision_1, recall_1, f1score_1))
#print(classification_report(y_test, y_pred))
print("<<Accuracy Rate: {0:.2f}%>>".format(pred*100))
