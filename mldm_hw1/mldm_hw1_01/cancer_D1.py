import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

print("===============KNN_Tuning===============")

best_score = 0;
best_param = {};
max_feature = []
grid_param = {
    'n_neighbors': [3, 4, 5, 6, 7]
}
combn = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], 7))
for i in combn:
    X_combn = X[:, i]
    X_combn_test = X_test[:, i]
    neigh = KNeighborsClassifier()
    gr = GridSearchCV(estimator=neigh, param_grid=grid_param, scoring='accuracy',
                      cv=5)
    gr.fit(X_combn, y)
    print(gr.best_params_, gr.best_score_)
    if best_score < gr.best_score_:
        best_score = gr.best_score_
        best_param = gr.best_params_
        max_feature = i

X_best = X[:, max_feature]
X_best_test = X_test[:, max_feature]
neigh = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'])
neigh.fit(X_best, y)
pred = neigh.score(X_best_test, y_test)

print("\n\n<<\K value : {}>>".format(best_param))
print("<<Feature Used: ", max_feature, ">>")
print("<<Best Accuracy for training: {0:.2f}% >>".format(best_score))
print("\n\n<<Accuracy Rate: {0:.2f}% >>"
      .format(pred * 100))