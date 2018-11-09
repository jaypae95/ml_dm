import pandas as pd
import numpy as np
from mldm_hw1_02 import knn


def my_confusion_matrix(y_test, y_pred):
    matrix = np.zeros((10, 10))
    for i in range(len(y_test)):
        matrix[y_test[i]][y_pred[i]] += 1  # (real, predict)

    return matrix


def my_classification_report(cm, len):
    precision = []
    recall = []
    f1_score = []

    for i in range(0, len):
        tp = cm[i][i]  # tp
        fn = np.sum(cm[i])-tp  # fn
        fp = np.sum(cm[:, i])-tp  # fp
        tn = np.sum(cm)-(tp+fn+fp)  # tn

        precision.append(round(tp/(tp+fp), 2))  # precision list
        recall.append(round(tp/(tp+fn), 2))  # recall list
        f1_score.append(round(
            2*precision[i]*recall[i]/(precision[i]+recall[i]), 2))  # f1_score list

    report = [precision, recall, f1_score]
    report = np.array(report)
    return report


# open file
with open("digits_train.csv", 'r') as trainfile:
    df = pd.read_csv(trainfile, names=[x for x in range(0, 785)])
with open("digits_test.csv", 'r') as testfile:
    test = pd.read_csv(testfile, names=[x for x in range(0, 785)])

# split label from data
X = np.array(df.drop(0, axis=1).values)
y = np.array(df[0].values)

X_test = np.array(test.drop(0, axis=1).values)
y_test = np.array(test[0].values)


fit = knn.train(k=4, distance='euclidean', cv=0)    # euclidean
# fit = knn.KNNClassifier(k=4, distance='manhattan')     # manhattan
# fit = knn.KNNClassifier(k=4, distance='l_infinity')    # L_infinity
train = fit(X, y)

print("="*25, "Test with Train Data", "="*25)
score, y_pred = knn.predict(X, y, train)      # test the train data

cm = my_confusion_matrix(y, y_pred)    # confusion matrix
report = my_classification_report(cm, 10).transpose()   # report

report = pd.DataFrame(report, columns=['Precision', 'Recall', 'F1-Score'])  # change to data frame

print("\nTrain Confusion Matrix \n", cm)
print("\nTrain Report\n", report)
print("<<Train Accuracy Rate: {0:.2f}%>>"
      .format(score*100))

print("="*25, "Test with Test Data", "="*25)
score, y_pred = knn.predict(X_test, y_test, train)      # test the test data

cm = my_confusion_matrix(y_test, y_pred)    # confusion matrix
report = my_classification_report(cm, 10).transpose()   # report

report = pd.DataFrame(report, columns=['Precision', 'Recall', 'F1-Score'])  # change to data frame

print("\nTest Confusion Matrix \n", cm)
print("\nTest Report\n", report)
print("<<Test Accuracy Rate: {0:.2f}%>>"
      .format(score*100))
