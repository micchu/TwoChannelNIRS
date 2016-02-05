#coding:utf-8

'''
Created on 2016/02/04
@author: misato

Signal SlopeとSignal Meanの特徴量の抽出

'''

import numpy as np

def get_SS_SM(signal, start, window):
    """
    Signal Slope(信号の傾き)とSignal Mean(信号平均)の特徴量の抽出
    @param signal: 信号 array型
    @param start: 計算開始点 (Naseerによると2秒目)
    @param window: 計算点幅  (Naseerによると5秒経過後) 
    @return sinal slope, signal mean
    
    動作確認済
    """
    # 計算対象領域の抽出
    target_signal = signal[start:start+window]
#    print target_signal
    
    # Signal Slopeの計算
    slope, seppen = np.polyfit(target_signal, range(len(target_signal)), 1)
    ss = slope
    
    # Signal Meanの計算
    sm = np.average(target_signal)
    
    return ss, sm

if __name__=="__main__":
    import random
    robj = random.Random()
    robj.seed(100)
    signal = [robj.random() for _ in range(100)]
    start = 10
    window = 50
    for s in signal:
        print s
    ss, sm = get_SS_SM(signal, start, window)
    print 
    print ss
    print sm