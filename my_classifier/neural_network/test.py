#coding:utf-8

'''
Created on 2016/01/28
@author: misato

元ファイルは田村さん作
関数やパラメータを整理して整形

Update on 2016/02/01
回帰タイプNNを追加
'''

import numpy as np
from chainer import cuda

from forward import forward_data, forward_data_regression

def test_nn_class(model, x_test, y_test):
    """
    NNを学習させる
    Adamと呼ばれるパラメータの最適化手法を使用
    @param model: NNの構造モデルオブジェクト
    @param x_test: テストデータの特徴量
    @param y_test: テストデータの教師信号
    @return: 識別率、ロス、識別結果のリスト、ロスのリスト、識別結果の確信度のリスト
    """
    # テストサンプル数
    test_sample_size = len(x_test)
    sum_loss = 0
    sum_accuracy = 0
    
    # 実際の出力クラスとその確信度
    result_class_list = []
    result_loss_list = []
    result_class_power_list = []

    for i in xrange(0, test_sample_size):
        x_batch = x_test[i:i+1]
        y_batch = y_test[i:i+1]
        # 順伝播させて誤差と精度を算出
        loss, acc, output= forward_data(model, x_batch, y_batch, train=False)

        # 結果の格納
        result_class_list.append(np.argmax(output.data))
        result_loss_list.append(float(cuda.to_cpu(loss.data)))
        result_class_power_list.append(output.data)
        sum_loss += float(cuda.to_cpu(loss.data))
        sum_accuracy += float(cuda.to_cpu(acc.data))
    
    loss = sum_loss / test_sample_size
    accuracy = sum_accuracy / test_sample_size
    
    return accuracy, loss, result_class_list, result_loss_list, result_class_power_list

def test_nn_regression(model, x_test, y_test):
    """
    回帰タイプのNNを学習させる
    @param model: NNの構造モデルオブジェクト
    @param x_test: テストデータの特徴量
    @param y_test: テストデータの教師信号
    @return: 予測結果のリスト、ロスのリスト
    """
    # テストサンプル数
    test_sample_size = len(x_test)
    sum_loss = 0
    
    # 実際の出力クラスとその確信度
    predicted_value_list = []
    loss_list = []
     
    for i in xrange(0, test_sample_size):
        x_batch = x_test[i:i+1]
        y_batch = y_test[i:i+1]
        
        # 順伝播させて誤差と予測値を算出
        loss, predicted = forward_data_regression(model, x_batch, y_batch, train=False)

        # 結果の格納
        predicted_value_list.append(float(cuda.to_cpu(predicted.data)))
        loss_list.append(float(cuda.to_cpu(loss.data)))
        sum_loss += float(cuda.to_cpu(loss.data))
    
    return predicted_value_list, loss_list
    
    