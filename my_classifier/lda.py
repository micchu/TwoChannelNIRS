#coding:utf-8

'''
Created on 2016/02/23

@author: misato
'''

import numpy as np
import sklearn.lda as slda
import sklearn.metrics as smet
from sklearn.multiclass import OneVsRestClassifier

def binary_lda(training_feature_array, training_label_array,
               test_feature_array, test_label_array):
    """
    2クラス分類LDA（しかし多クラス分類で代用可能だった現実...）
    @param training_feature_array: トレーニング用データ
    @param training_label_array: トレーニング用データラベル
    @param test_feature_array: テスト用データ
    @param test_label_array: テスト用データラベル
    
    @return: 識別率, 識別結果のリスト, 識別されたクラスへの所属確率のリスト, LDAオブジェクト
    
    動作確認済
    """
    lda_obj = slda.LDA()
    lda_obj.fit(training_feature_array, training_label_array)
    
    print "test..."
    class_result = lda_obj.predict(test_feature_array)
    proba_result = lda_obj.predict_proba(test_feature_array)
    proba_max_result = np.max(proba_result, axis=1)
    try:
        precision = smet.accuracy_score(test_label_array, class_result)
        #precision, recall, fmeasure, sup = smet.precision_recall_fscore_support(test_label_array, class_result, 
        #                                                                        average='micro')    
    except DeprecationWarning, e:
        pass
    
    return precision, class_result, proba_max_result, lda_obj


def multiclass_lda(training_feature_array, training_label_array,
               test_feature_array, test_label_array):
    """
    多クラス分類LDA
    @param training_feature_array: トレーニング用データ
    @param training_label_array: トレーニング用データラベル
    @param test_feature_array: テスト用データ
    @param test_label_array: テスト用データラベル
    
    @return: 全体識別率, 識別結果のリスト, 識別されたクラスへの所属確率のリスト, LDAオブジェクト

    動作確認済
    """
    multi_lda_obj = OneVsRestClassifier(slda.LDA())
    multi_lda_obj.fit(training_feature_array, training_label_array)
    
    print "test..."
    class_result = multi_lda_obj.predict(test_feature_array)
    proba_result = multi_lda_obj.predict_proba(test_feature_array)    
    proba_max_result = np.max(proba_result, axis=1)
    try:
        precision = smet.accuracy_score(test_label_array, class_result)
    except DeprecationWarning, e:
        pass

    return precision, class_result, proba_max_result, multi_lda_obj


if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    def plot_data_plane(X, Y, test_feature_array, multi = True):    

        min_x = np.min(X[:, 0])
        max_x = np.max(X[:, 0])
        min_y = np.min(X[:, 1])
        max_y = np.max(X[:, 1])

        plt.scatter(X[:, 0], X[:, 1], s=40, c='blue')
        plt.scatter(test_feature_array[:, 0], test_feature_array[:, 1], s=40, c='red')
        
        if multi:
            plot_hyperplane(lda_obj.estimators_[0], min_x, max_x, 'r--',
                            'Boundary\nfor class 1')
            plot_hyperplane(lda_obj.estimators_[1], min_x, max_x, 'k-.',
                            'Boundary\nfor class 2')
            plot_hyperplane(lda_obj.estimators_[2], min_x, max_x, 'b-.',
                            'Boundary\nfor class 3')
        else:
            plot_hyperplane(lda_obj, min_x, max_x, 'r--',
                            'Boundary\nfor class 1')
            
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axhline(linewidth=1.7, color="k")
        plt.axvline(linewidth=1.7, color="k")
        plt.xticks(())
        plt.yticks(())
    
        plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
        plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    
        plt.figure(figsize=(8, 6))
        plt.show()

    def plot_hyperplane(clf, min_x, max_x, linestyle, label):
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -1 * w[0] / w[1]
        xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
        yy = a * xx - (clf.intercept_[0]) / w[1]
        plt.plot(xx, yy, linestyle, label=label)    
        
        
    if True:
        training_feature_array = np.array([[1,1],[1,0.5],[0.5,1],[0,0]], dtype=np.float32)
        training_label_array = np.array([1,1,0,0], dtype=np.float32)
        test_feature_array = np.array([[1,0.8],[0.7,0.5],[0.2,0.2]], dtype=np.float32)
        test_label_array = np.array([1,0,0], dtype=np.float32)
        precision, class_result, proba_result, lda_obj = multiclass_lda(training_feature_array, training_label_array, test_feature_array, test_label_array)
#        precision, class_result, proba_result, lda_obj = binary_lda(training_feature_array, training_label_array, test_feature_array, test_label_array)
        print precision
        print class_result
        print proba_result
        plot_data_plane(training_feature_array, training_label_array, test_feature_array, multi = False)
        
    if False:
        training_feature_array = np.array([[1,2],[2,1],[2,3],[3,3],[3,1],[3,2]], dtype=np.float32)
        training_label_array = np.array([0,0,1,1,2,2], dtype=np.float32)
        test_feature_array = np.array([[0,0],[2.5,3],[0.2,0.2],[3,1.2]], dtype=np.float32)
        test_label_array = np.array([0,1,2,2], dtype=np.float32)
        precision, class_result, proba_result, lda_obj = multiclass_lda(training_feature_array, training_label_array, test_feature_array, test_label_array)
        print precision
        print class_result
        print proba_result
        plot_data_plane(training_feature_array, training_label_array, test_feature_array)
    

                        