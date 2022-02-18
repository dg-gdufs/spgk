'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-27 20:48:11
LastEditors: Renhetian
LastEditTime: 2022-02-18 20:16:45
'''

from codes.Utils import time_now_formate
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics


class SVMClassification:

    save_path = 'model/svm/'

    def __init__(self, loader, cv=10) -> None:
        self.loader = loader
        self.cv = cv
        self.save_path += loader.dataset_name

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):

        feature = self.loader.kernel_matrix
        label = self.loader.label

        X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=.3,random_state=0)
        # 训练模型
        model = OneVsRestClassifier(SVC(kernel='rbf',probability=True,random_state=0))
        print("[INFO] Successfully initialize a new model !")
        print("[INFO] Training the model…… ")
        clt = model.fit(X_train,y_train)
        print("[INFO] Model training completed !")
        # 保存训练好的模型，下次使用时直接加载就可以了
        joblib.dump(clt, self.save_path + '/model.pkl')
        print("[INFO] Model has been saved !")
        '''
        # 加载保存的模型
        clt = joblib.load("D:/spg/spgk/SVM.pkl")
        print("model has been loaded !")
        # y_train_pred = clt.predict(X_train)
        '''
        y_test_pred = clt.predict(X_test)
        ov_acc = metrics.accuracy_score(y_test_pred,y_test)
        print("overall accuracy: %f"%(ov_acc))
        print("===========================================")
        acc_for_each_class = metrics.precision_score(y_test,y_test_pred,average=None)
        print("acc_for_each_class:\n",acc_for_each_class)
        print("===========================================")
        avg_acc = np.mean(acc_for_each_class)
        print("average accuracy:%f"%(avg_acc))
        print("===========================================")
        classification_rep = classification_report(y_test,y_test_pred,
                                                target_names=None)
        print("classification report: \n",classification_rep)
        print("===========================================")

        print("[INFO] Successfully get SVM's classification overall accuracy ! ")