#coding:utf-8

'''
Created on 2016/01/29
@author: misato

動作検証済

Update on 2016/01/30
ソートする前に一度ランダマイズをする機構を追加
ソートにラベルのみをキーとするように指定
'''

import numpy as np

def separate_dataset_for_K_FOLD(label_list, feature_array, num, randobj):
    """
    データセットのk-fold用に分割
    @param label_list: ラベルのリスト
    @param feature_array: 特徴量のリスト 二次元array型 （サンプル数×特徴次元数）
    @param num: 分割数 
    @param randobj: 乱数オブエジェクト
    
    @return: 分割されたデータセット
        dataset_label_list ラベルのリスト
        dataset_feature_array 特徴量の3次元array型 データセット分割数×サンプル数×特徴次元数
    """
    # データの学習セットの構築（できれば保存）
    
    # ランダマイズしてから、ラベル順にソート
    # ラベルとindexを組にして操作
    dataset = zip(label_list, range(len(label_list)))    
    randobj.shuffle(dataset)
    randobj.shuffle(dataset)
    dataset.sort(key=lambda x: x[0])
    
    dataset_feature_list = [[] for _ in range(num)]
    dataset_label_list = [[] for _ in range(num)]
    counter = 0
    for label, index in dataset:
        dataset_label_list[counter].append(label)
        dataset_feature_list[counter].append(feature_array[index])
        counter += 1
        if counter == num:
            counter = 0
    
    # array型への変換
    dataset_feature_array = np.asarray(dataset_feature_list)
    
    return dataset_label_list, dataset_feature_array

if __name__=="__main__":
    import numpy as np
    import random as rr
    a = list(np.tile([1,2,3,4,5], 5))
    print a 
    b = list([[1+i,2+i] for i in range(25)])
    ret = separate_dataset_for_K_FOLD(a, b, 3, rr)
    print ret[0]
    for r in  ret[1]:
        print r
    
    