#coding:utf-8

'''
Created on 2016/01/28
@author: misato
'''

import chainer.functions  as F
from chainer import FunctionSet

def set_model(feature_dimension_size, param, output_units):
    """
    NNのモデルの設定
    @param feature_dimension_size: 入力特徴量の次元数
    @param param: パラメータオブジェクト  
    @param output_units: 出力層のユニット数 

    @return: ネットワークモデル
    """
    n_units   = param.NODE_NUM_1LAYER
    model = FunctionSet(l1=F.Linear(feature_dimension_size, n_units),
                l2=F.Linear(n_units, n_units),
                l3=F.Linear(n_units, output_units))
    return model

