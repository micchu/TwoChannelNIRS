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
    if param.HIDDEN_LAYER_NUM == 1:
        # 3層パーセプトロン（入力層、中間層1、出力層）
        model = FunctionSet(l1=F.Linear(feature_dimension_size, param.NODE_NUM_1LAYER),
                    l2=F.Linear(param.NODE_NUM_1LAYER, output_units))
    if param.HIDDEN_LAYER_NUM == 2:
        # 4層パーセプトロン（入力層、中間層1、中間層2、出力層）
        model = FunctionSet(l1=F.Linear(feature_dimension_size, param.NODE_NUM_1LAYER),
                    l2=F.Linear(param.NODE_NUM_1LAYER, param.NODE_NUM_2LAYER),
                    l3=F.Linear(param.NODE_NUM_2LAYER, output_units))
    if param.HIDDEN_LAYER_NUM == 3: 
        # 5層パーセプトロン
        model = FunctionSet(l1=F.Linear(feature_dimension_size, param.NODE_NUM_1LAYER),
                    l2=F.Linear(param.NODE_NUM_1LAYER, param.NODE_NUM_2LAYER),
                    l3=F.Linear(param.NODE_NUM_2LAYER, param.NODE_NUM_3LAYER),
                    l4=F.Linear(param.NODE_NUM_3LAYER, output_units))
    return model

