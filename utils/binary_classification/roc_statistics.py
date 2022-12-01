from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression

def calculate_roc_auc_scores(groundtruth, prediction, filename_and_path):
    fpr, tpr, _ = roc_curve(groundtruth, prediction)
    #print("FPR: {}\nTPR: {}".format(fpr, tpr))
    roc_auc = auc(fpr, tpr)
    print("AUROC: {}".format(roc_auc))
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(filename_and_path)


if __name__ == '__main__':
    X_train = np.concatenate((np.random.normal(10, 15, 10000), np.random.normal(16, 10, 10000)))
    X_train = X_train[:, np.newaxis]
    Y_train = np.concatenate((np.zeros(10000), np.ones(10000))).astype(int)
    X_test = np.concatenate((np.random.normal(10, 15, 10000), np.random.normal(16, 10, 10000)))
    X_test = X_test[:, np.newaxis]
    Y_test = np.concatenate((np.zeros(10000), np.ones(10000))).astype(int)

    classifier = SVC(kernel='linear', probability=True)
    y_pred = classifier.fit(X_train, Y_train).decision_function(X_test)
    print('Accuracy of SVM classifier on training set: {:.2f}'
          .format(classifier.score(X_train, Y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
          .format(classifier.score(X_test, Y_test)))
    calculate_roc_auc_scores(Y_test, y_pred, 'roc_auc_test.png')
    a=1