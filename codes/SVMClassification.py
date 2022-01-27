'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-27 20:48:11
LastEditors: Renhetian
LastEditTime: 2022-01-27 21:17:17
'''

from codes.Utils import time_now_formate
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedShuffleSplit


class SVMClassification:

    save_path = 'model/svm/'

    def __init__(self, loader, cv=10) -> None:
        self.loader = loader
        self.cv = cv
        self.save_path += loader.dataset_name

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        """
        Given a kernel matrix, performs 10-fold cross-validation using an SVM and returns classification accuracy.
        At each iteration the optimal value of parameter C is determined using again cross-validation.
        """
        K = self.loader.kernel_matrix
        labels = self.loader.label

        print("\nStarted 10-fold cross validation:")

        # Number of folds
        cv = self.cv

        # Specify range of C values
        C_range = 10. ** np.arange(1,5,1)

        # Output variables
        result = {}
        result["opt_c"] = np.zeros(cv)
        result["accuracy"] = np.zeros(cv)
        result["f1_score"] = np.zeros(cv)

        kf = KFold(n_splits=cv, shuffle=True, random_state=None)

        # Current iteration
        iteration = 0

        #Perform k-fold cv
        for train_indices_kf, test_indices_kf in kf.split(K,labels):
            
            labels_current = labels[train_indices_kf]
            
            K_train = K[np.ix_(train_indices_kf, train_indices_kf)]
            labels_train = labels[train_indices_kf]

            K_test = K[np.ix_(test_indices_kf, train_indices_kf)]
            labels_test = labels[test_indices_kf]

            # Optimize parameter C
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=None)
            for train_index, test_index in sss.split(K_train, labels_train):
                K_C_train = K[np.ix_(train_index, train_index)]
                labels_C_train = labels[train_index]

                K_C_test = K[np.ix_(test_index, train_index)]
                labels_C_test = labels[test_index]

                best_C_acc = -1
                for i in range(C_range.shape[0]):
                    C = C_range[i]
                    clf = SVC(C=C,kernel='precomputed')
                    clf.fit(K_C_train, labels_C_train) 
                    labels_predicted = clf.predict(K_C_test)
                    C_acc = accuracy_score(labels_C_test, labels_predicted)
                    if C_acc > best_C_acc:
                        best_C_acc = C_acc
                        result["opt_c"][iteration] = C

            clf = SVC(C=result["opt_c"][iteration],kernel='precomputed')
            clf.fit(K_train, labels_train) 
            labels_predicted = clf.predict(K_test)
            result["accuracy"][iteration] = accuracy_score(labels_test, labels_predicted)
            result["f1_score"][iteration] = precision_recall_fscore_support(labels_test, labels_predicted, pos_label=None, average='macro')[2]
            iteration += 1
            print("Iteration " + str(iteration) + " complete")

        max_accuracy = -1
        opt_iteration = 0
        for i in range(iteration):
            if result["accuracy"][i] > max_accuracy:
                max_accuracy = result["accuracy"][i]
                opt_iteration = i
        clf = SVC(C=result["opt_c"][opt_iteration],kernel='precomputed')
        clf.fit(K, labels)

        result["mean_accuracy"] = np.mean(result["accuracy"]) 
        result["mean_f1_score"] = np.mean(result["f1_score"])
        result["std"] = np.std(result["accuracy"])

        print("\nAverage accuracy: ", result["mean_accuracy"])
        print("Average macro f1-score: ", result["mean_f1_score"])
        print("-------------------------------------------------")

        save_path = self.save_path + '/{}.pkl'.format(time_now_formate())
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)