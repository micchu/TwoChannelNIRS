#coding:utf-8

'''
Created on 2016/01/28
@author: misato

挙動確認済
'''

import numpy as np

def relabel_step(label_list, class_num):
    """
    固定値でリラベル
    現状5classのみ実装
    @param label_list: ラベルのリスト
    @param class_num: クラス数
    @return: 新しいラベルのリスト（順序は与えられたラベルのリストと同じ）  
    """
    new_label_list = []

    # 5クラスの場合
    if class_num == 5:
        for i in range(len(label_list)):
            new_label = int((label_list[i]-1)/2) 
#            new_label = label_mapping[new_label]
            new_label_list.append(new_label)
    
    return new_label_list
    
def relabel_minimize_variance(label_list, class_num):
    """
    分散最小化でリラベル（各クラスのサンプル数をできるだけ揃える）
    現状5クラスのみ実装
    @param label_list: ラベルのリスト
    @param class_num: クラス数
    @return: 新しいラベルのリスト（順序は与えられたラベルのリストと同じ）  
    """
    new_label_list = []
    sample_num = len(label_list)

    # 各クラスのラベル数をカウントし、クラスの昇順にソート
    distinct_class_label = sorted(set(label_list))
    distinct_class_num = len(distinct_class_label)
    
    label_num_list = [label_list.count(c) for c in distinct_class_label]
#    print distinct_class_label
#    print label_num_list
#    print 
    
    # 5クラスの場合
    if class_num == 5:
        # 分散最小の組合せを求める
        min_var = 1000
        min_parameter = []
        for i in range(1, distinct_class_num - 3):
            a = sum(label_num_list[:i])
            for j in range(i+1, distinct_class_num - 2):
                b = sum(label_num_list[i:j])
                for k in range(j+1, distinct_class_num - 1):
                    c = sum(label_num_list[j:k])
                    for l in range(k+1, distinct_class_num):
                        d = sum(label_num_list[k:l])
                        e = sum(label_num_list[l:])
                        
                        variance = np.var([a,b,c,d,e])
                        if variance < min_var:
#                            print min_var, i, j, k, l
                            min_var = variance
                            min_parameter = [i, j, k, l]
        # クラスのマッピングの作成
        i,j,k,l = min_parameter
#        print "result", min_var, i,j,k,l
#        print label_num_list[:i]
#        print label_num_list[i:j]
#        print label_num_list[j:k]
#        print label_num_list[k:l]
#        print label_num_list[l:]
        class_mapping = [0]
        class_mapping.extend([1 for _ in range(i)])
        class_mapping.extend([2 for _ in range(j-i)])
        class_mapping.extend([3 for _ in range(k-j)])
        class_mapping.extend([4 for _ in range(l-k)])
        class_mapping.extend([5 for _ in range(distinct_class_num-l)])
#        print "mapping", class_mapping
        
        # ラベルの置き換え
        for i in range(len(label_list)):
            new_label_list.append(class_mapping[label_list[i]])

        return new_label_list
        
if __name__=="__main__":
    import random
    label_list = [random.randint(1,10) for i in range(40)]
    print label_list
    _label_list = sorted(label_list)
    new_label_list = relabel_step(label_list, 5)
    #new_label_list = relabel_minimize_variance(_label_list, 5)    
#    print _label_list
    print new_label_list
    
