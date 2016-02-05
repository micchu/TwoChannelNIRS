#coding:utf-8

'''
Created on 2016/02/01
@author: misato

回帰計算における評価
'''

import scipy
import scipy.stats as scst
import scipy.spatial.distance as spdi
import numpy as np

def get_loss_per_class(label_list, test_label_list, loss_list):
    """
    クラス毎のロスを計算
    @param label_list: ラベルのリスト
    @param test_label_list: 教師信号のリスト 
    @param loss_list: ロスのリスト
    
    @return クラス毎のロスリスト
    
    動作確認済
    """
    loss_list_per_class = {label:[] for label in label_list}
    #print loss_list_per_class
    for i in range(len(test_label_list)):
        loss_list_per_class[test_label_list[i]].append(loss_list[i])
    loss_per_class = []
    #print loss_list_per_class
    for label in label_list:
        if len(loss_list_per_class[label]) == 0:
            loss_per_class.append(NaN)
        else:
            loss_per_class.append(np.average(loss_list_per_class[label]))
    print loss_per_class
    
    return loss_per_class 

def get_correlation(label_list, result_list):
    """
    ピアソンの相関係数を算出
    @param label_list: ラベルのリスト 
    @param result_list: 出力結果のリスト
    @return 相関係数, p_value
    
    Spearmanは順位尺度対象ということで、間隔尺度対象のピアソンを採用した。
    しかし、ピアソンは正規分布が前提にあり（平均値からの差をかけたり割ったりするから）、これもうまくいかぬ
    距離計算を提案する
    
    動作確認済
    """
    # 相関係数とp-valueを算出
#    rho, p_value = scst.spearmanr(label_list, result_list)
    rho, p_value = scst.pearsonr(label_list, result_list)
    
    return rho, p_value

def get_distance(label_list, result_list):
    """
    ユークリッド距離を計算...なんかロスと一緒じゃね？→一緒じゃなかった...何故だろう...
    @param label_list: ラベルのリスト 
    @param result_list: 出力結果のリスト
    @return: 距離総和
    """
    ret = spdi.euclidean(label_list, result_list)
    average_ret = ret/len(label_list)
    return average_ret

if __name__=="__main__":
    a = [1,2,3,4]
    b = [2,3,4,5]
    #print get_spearman(a, b)
    label_list = [1,2,3]
    test_label_list = [1,2,3,1,2,3]
    loss_list = [3,4,5,3,4,5]
    print get_loss_per_class(label_list, test_label_list, loss_list)