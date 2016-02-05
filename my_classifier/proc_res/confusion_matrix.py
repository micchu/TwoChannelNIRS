#coding:utf-8

'''
Created on 2016/01/29

@author: misato
'''

import numpy as np

def calculate_accuracy_per_label(label_list, confusion_mat):
    """
    混同行列をクラス毎に処理
    @param label_list: クラスのリスト
    @param confusion_mat: 混同行列  
    @return: 平均適合率、平均再現率、平均f-Meature、クラス*(適合率、再現率、f-Meature)
    """
    result_mat = np.zeros((len(label_list), 3))
    
    # 各クラスの計算 devision errorに注意
    for i in range(len(label_list)):
        # クラスの適合率
        if sum(confusion_mat[:,i]) == 0:
            result_mat[i][0] = 0.0
        else:
            result_mat[i][0] = confusion_mat[i,i] / sum(confusion_mat[:,i])
        # クラスの再現率 
        if sum(confusion_mat[i,:])  == 0:
            result_mat[i][1] = 0.0
        else:
            result_mat[i][1] = confusion_mat[i,i] / sum(confusion_mat[i,:]) 
        # クラスのF-measure
        if (result_mat[i][0] + result_mat[i][1]) == 0:
            result_mat[i][2] = 0.0
        else:
            result_mat[i][2] = (2 * result_mat[i][0] * result_mat[i][1]) / (result_mat[i][0] + result_mat[i][1])
    
    # 平均算出
    precision = np.average(result_mat[:, 0])
    recall = np.average(result_mat[:, 1])
    f_measure = np.average(result_mat[:, 2])
    
    return precision, recall, f_measure, result_mat

def get_confusion_matrix(label_list, test_label_list, result_label_list):
    """
    混同行列の生成
    @param label_list: クラスリスト
    @param test_label_list: 教師信号
    @param result_label_list: 実際の識別結果
    """
    # 混同行列のベース作成
    mat = np.zeros((len(label_list), len(label_list)))
    # クラス名とインデックスのマッピング
    label_mapping = {}
    for i in range(len(label_list)):
        label_mapping[label_list[i]] = i
    # 振り分け
    for i in range(len(test_label_list)):
        test_index = label_mapping[test_label_list[i]]
        result_index = label_mapping[result_label_list[i]]
        mat[test_index, result_index] += 1
    return mat


if __name__=="__main__":
    #  get_confusion_matrix 動作検証
    if False:
        label_list = np.array([1,2,3,4,5])
        test_label_list = np.tile(label_list, 3)
        result_label_list = np.tile(label_list, 3)
        ret = get_confusion_matrix(label_list, test_label_list, result_label_list)
        print test_label_list
        print result_label_list
        print ret

    # calculate_accuracy_per_labelの動作検証 
    if True:
        label_list = np.array([1,2,3,4,5])
        test_label_list = np.tile(label_list, 3)
        result_label_list = np.tile(label_list, 3)
        result_label_list[0:5] -= 1
        result_label_list[0] += 2 
        ret = get_confusion_matrix(label_list, test_label_list, result_label_list)
        print test_label_list
        print result_label_list
        print ret
        res = calculate_accuracy_per_label(label_list, ret)
        print "Precision", "Recall", "F-Measure"  
        print res 
