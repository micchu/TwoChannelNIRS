#coding:utf-8

'''
Created on 2016/02/04
@author: misato
'''

import numpy as np
import itertools

from my_error import MyError
import external_library.smote as ext_smote

def smote(feature_list, new_sample_num, resampling_size):
    """
    SMOTEによるリサンプリング
    指定された数のサンプルを作成し、返す
    なお、無作為抽出のみ実装している
    @param feature_list: あるクラスの特徴量のリスト リストの中身はarrayになっている
    @param new_sample_num: 作成するサンプル数
    @param resampling_size: リサンプリングに使う数

    @return 新しいサンプルのリスト list型 配列の中身はarray型

    bootstrapと異なり、duplicationやrandomオブジェクトはなし、選択はsmoteライブラリ内で行われるので 
    """
    # 与えられたサンプル数がリサンプリング手法での抽出数に満たない場合
    if len(feature_list) < resampling_size:
        raise MyError("Sample Size is smaller than the number of resampling size.")
    
    # 一旦array型に変換
    feature_array = np.asarray(feature_list, dtype=np.float32)
    # SMOTEを実行
    percentage_new_sample = int(round(float(new_sample_num) / len(feature_array) * 100))
    new_feature_array = ext_smote.SMOTE(feature_array, percentage_new_sample, resampling_size)
    
    # 生成された各サンプルをリストに追加して行く
    new_feature = []
    for i in range(new_sample_num):
        new_feature.append(new_feature_array[i])
        
    return new_feature

if __name__=="__main__":
    import random
    a = range(100)
    ar = np.reshape(a, (10,10))
    print ar
    
    ret = smote(ar, 10, 3, random)
    print len(ret)
    for r in ret:
        print r
