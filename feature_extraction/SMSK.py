#coding:utf-8

'''
Created on 2016/02/22

@author: misato
'''

import numpy as np

def get_SM_SK(signal):
    """
    Signal Mean(信号平均)とSignal Skewness(信号の不均衡度合)の算出
    @param signal: 信号 array型
    @return: signal mean, signal skewness
    
    動作確認済
    """
    # 計算対象領域の抽出
    target_signal = signal

    # Signal Meanの計算
    sm = np.average(signal)
    
    # Signal Skewness
    sigma = np.std(signal)
    new_signal = signal - sm
    #print new_signal
    new_signal /= sigma
    #print new_signal
    new_signal = np.power(new_signal, 3)
    #print new_signal
    sk = sum(new_signal)
    
    return sm, sk

if __name__=="__main__":
    data = np.array([1,2,3,7,9])
    print get_SM_SK(data)
    