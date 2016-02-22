#coding:utf-8

'''
Created on 2016/01/27
@author: misato

動作確認済

Update on 2016/01/30
抽出の組合せに重複なしのケースを実装
組合せの数がサンプルサイズを満たさない場合はエラーを発生させる
'''

import numpy as np
import itertools

from my_error import MyError

def bootstrap(feature_list, new_sample_num, resampling_size, randobj, duplication=True):
    """
    ブートストラップによるリサンプリング
    指定された数のサンプルを作成し、返す
    なお、無作為抽出のみ実装している
    @param feature_list: 特徴量のリスト 配列の中身はarrayになっている
    @param new_sample_num: 作成するサンプル数
    @param resampling_size: リサンプリングに使う数
    @param randobj: 乱数モジュール 
    @param duplication: 親の重複を許すか否か（defaultはTrue） 
    @return 新しいサンプルのリスト list型 配列の中身はarray型
    
    """
    # 与えられたサンプル数がリサンプリング手法での抽出数に満たない場合
    if len(feature_list) < resampling_size:
        raise MyError("Sample Size is smaller than the number of resampling size.")
    
    # 無作為抽出によるサンプルの生成
    new_feature_list = []
    index_list = range(len(feature_list))
    if duplication:
        for s in range(new_sample_num):
            randobj.shuffle(index_list)
            group_index_list = index_list[:resampling_size]
            #print group_index_list
            new_feature = np.average(feature_list[group_index_list], axis=0) # 行間で平均 @UndefinedVariable
            #print new_feature
            new_feature_list.append(list(new_feature))
    else:
        # 全ての組合せを生成
        combi = list(itertools.combinations(index_list, resampling_size))
        # 組合せの数が生成数に足りない場合
        if len(combi) < new_sample_num:
            raise MyError("Sample Combination Size is smaller than the number of resampling size.")
        # ランダマイズ
        randobj.shuffle(combi)

        # 必要なサンプル数を選択し、特徴量を平均
        for s in range(new_sample_num):
            new_feature = np.average([feature_list[i] for i in combi[s]], axis=0)    
            new_feature_list.append(list(new_feature))
        
    return new_feature_list

if __name__=="__main__":
    import random
    a = range(100)
    ar = np.reshape(a, (10,10))
    print ar
    
    ret = bootstrap(ar, 10, 3, random, duplication=False)
    for r in ret:
        print r