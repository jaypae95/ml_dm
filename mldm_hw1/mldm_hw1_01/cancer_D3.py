import pandas as pd
from sklearn.svm import SVC
from itertools import combinations
from sklearn.model_selection import GridSearchCV

with open("cancer_train.csv", 'r') as trainfile:
    df = pd.read_csv(trainfile, names=[x for x in range(0, 10)])
with open("cancer_test.csv", 'r') as testfile:
    test = pd.read_csv(testfile, names=[x for x in range(0, 10)])

X = df.drop(0, axis=1).values
y = df[0].values

X_test = test.drop(0, axis=1).values
y_test = test[0].values

print("===============SVM===============")

best_score = 0;
best_param = {}
max_feature = []

grid_param = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['auto', 0.01, 0.1, 1, 10]
}
print("Please Wait...\n\n")
for com_count in range(9, 7, -1):
    combn = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], com_count))  # feature used
    for i in combn:
        X_combn = X[:, i]
        X_combn_test = X_test[:, i]
        clf = SVC()
        # search best parameters with cross validation using grid search
        gr = GridSearchCV(estimator=clf, param_grid=grid_param, scoring='accuracy',
                          cv=5)
        gr.fit(X_combn, y)
        print(i, "//// ", gr.best_params_, "///// {0:.1f}%".format(gr.best_score_*100))
        if best_score < gr.best_score_:
            best_score = gr.best_score_
            best_param = gr.best_params_
            max_feature = i

print("\n\n<<Kernel Used: {} >>".format(best_param['kernel']))
print("<<C Used: {} >>".format(best_param['C']))
print("<<Gamma Used: {} >>".format(best_param['gamma']))
print("<<Feature Used: {} >>".format(max_feature))
print("<<Best Accuracy for training: {0:.2f}% >>".format(best_score*100))
# test with best parameters
X_best = X[:, max_feature]
X_best_test = X_test[:, max_feature]
clf = SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
clf.fit(X_best, y)

pred = clf.score(X_best_test, y_test)

print("\n\n<<Test Accuracy Rate: {0:.2f}%>>"
      .format(pred * 100))

