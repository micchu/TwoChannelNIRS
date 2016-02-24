#coding:utf-8

'''
Created on 2016/02/24
@author: misato

参考URL

'''

import numpy as np
import sklearn.svm as sksvm
import sklearn.metrics as skmet
import sklearn.cross_validation as skcr
from sklearn.multiclass import OneVsRestClassifier

from my_error import MyError

def multiclass_svm(training_feature_array, training_label_array, test_feature_array, test_label_array, 
        kernel_type = "rbf", grid_search = True, n_fold = 5, costs = None, gammas = None):
    """
    多クラス分類のSVC
    @param training_feature_array: トレーニング用データ
    @param training_label_array: トレーニング用データラベル
    @param test_feature_array: テスト用データ
    @param test_label_array: テスト用データラベル
    @keyword kernel_type: カーネル種別
    @keyword grid_search: パラメータ最適化をするか否か 
    @keyword n_fold: フォールド数
    @keyword costs: コスト値リスト
    @keyword gammas: ガンマ値リスト  
    
    @return: 識別率, 識別結果のリスト, 識別面からの距離のリスト, SVCオブジェクト
    """
    # 多クラス識別器の生成
    multi_svm_model = OneVsRestClassifier(sksvm.SVC(kernel=kernel_type, probability=True))
#    print multi_svm_model.get_params()
    if grid_search:
        # パラメータ最適化
        ret_c, ret_gamma = optimizeParameter(multi_svm_model, kernel_type, training_feature_array, training_label_array, _fold=n_fold, _costs=costs, _gammas=gammas)
    else:
        # パラメータ最適化を行わない場合、一般的なデフォルト値を用いる
        # コスト値は1.0で、ガンマ値は1/特徴量次元数
        ret_c = 1.0
        ret_gamma = 1/len(training_feature_array[0,])
    # 最適なコスト値、ガンマ値の設定
    #multi_svm_model.set_params(C=ret_c, gamma=ret_gamma)
    multi_svm_model.estimator.set_params(C=ret_c, gamma=ret_gamma)
    
    # 学習
    multi_svm_model.fit(training_feature_array, training_label_array)
    # 予測
    result_class_list = multi_svm_model.predict(test_feature_array)
    # クラス尤度の計算
    result_probability_list = multi_svm_model.predict_proba(test_feature_array)
    
    # 識別率の計算
    try:
        precision = skmet.accuracy_score(test_label_array, result_class_list)
    except DeprecationWarning, e:
        pass
    
    return precision, result_class_list, result_probability_list, multi_svm_model
    
def optimizeParameter(multi_svm_model, kernel_type, feature_array, label_array, _fold = 5, _costs = None, _gammas = None):
    """
    Grid Searchによるパラメータ最適化
    @param multi_svm_model: SVMの多クラス用モデル
    @param kernel_type: カーネル種別  
    @param feature_array: 特徴量
    @param label_array: ラベル  
    @keyword _fold: フォールド数
    @keyword _costs: コストのリスト(default = 2^-5, 2^-3, ..., 2^13, 2^15)
    @param _gammas: ガンマのリスト(default = 2^3, 2^1, ..., 2^-13, 2^-15)   
    
    @return: 最適化済コスト値、ガンマ値
    
    コストとガンマはlibSVMを参照した
    -log2c {begin,end,step | "null"} : set the range of c (default -5,15,2)
    begin,end,step -- c_range = 2^{begin,...,begin+k*step,...,end}
    "null"         -- do not grid with c
    -log2g {begin,end,step | "null"} : set the range of g (default 3,-15,-2)
    begin,end,step -- g_range = 2^{begin,...,begin+k*step,...,end}
    "null"         -- do not grid with g
    """
    # クラスの適合性Check
    if type(multi_svm_model) != OneVsRestClassifier:
        raise MyError("SVM model's type is invalid. It must be OneVsRestClassifier.")
    
    # コストとガンマのdefault値の設定
    if _costs == None:
        _costs = [2**i for i in range(-5, 17, 2)]
    if _gammas == None:
        _gammas = [2**i for i in range(3, -17, -2)]
    # 線形の場合はガンマ値を0にしておく
    if kernel_type == "linear":
        _gammas = [1.0]
    
    best_c = _costs[0]
    best_gamma = _gammas[0]
    best_acc = 0
    for c in _costs:
        for g in _gammas:
            # コスト値とガンマ値のセット
            # 多クラス分類器(OneVsRestClassifier)にセットしているSVCに、estimator変数でアクセスする
            multi_svm_model.estimator.set_params(C=c, gamma = g)
            # クロスバリデーション（N-fold数分の識別率が返ってくる）
            acc_list = skcr.cross_val_score(multi_svm_model, feature_array, label_array, cv = _fold)
            acc = np.average(acc_list)
            # 最大の識別率を確保
            if acc > best_acc:
                best_c = c
                best_g = g
                best_acc = acc
#                print multi_svm_model.estimator.get_params()
#                print multi_svm_model.estimators_               
                
    return best_c, best_g

#def Linear_SVC(training_feature_array, training_label_array, test_feature_array, test_label_array, costs = None):
#    """
#    未実装
#    多クラス分類の線形SVC
#    @param training_feature_array: トレーニング用データ
#    @param training_label_array: トレーニング用データラベル
#    @param test_feature_array: テスト用データ
#    @param test_label_array: テスト用データラベル
#
#    @return: 識別率, 識別結果のリスト, 識別されたクラスへの所属確率のリスト, LDAオブジェクト 
#    """
#    svm_model = svm.LinearSVC()
    
if __name__=="__main__":
    if False:
        # Grid SearchのCheck
        training_feature_array = np.array([[1,2],[2,1],[2,3],[3,3],[3,1],[3,2]], dtype=np.float32)
        training_label_array = np.array([0,0,1,1,2,2], dtype=np.float32)
        multi_svm_model = OneVsRestClassifier(sksvm.SVC())        
        print optimizeParameter(multi_svm_model, "linear", training_feature_array, training_label_array, _fold=2)
        
    if True: 
        # SVMのCheck
        training_feature_array = np.array([[1,2],[2,1],[2,3],[3,3],[3,1],[3,2]], dtype=np.float32)
        training_label_array = np.array([0,0,1,1,2,2], dtype=np.float32)
        test_feature_array = np.array([[0,0],[2.5,3],[0.2,0.2],[3,1.2]], dtype=np.float32)
        test_label_array = np.array([0,1,2,2], dtype=np.float32)

        ret = multiclass_svm(training_feature_array, training_label_array, test_feature_array, test_label_array,
            kernel_type = "linear", grid_search = True, n_fold = 2, costs = None, gammas = None)
        print ret[0]
        print ret[1]
        print ret[2]
        print ret[3]
            