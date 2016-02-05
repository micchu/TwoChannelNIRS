#coding:utf-8

'''
Created on 2016/01/27
@author: misato

Update on 2016/02/01
スムージングする領域を修正
[0,a,a,tar,a,0,0...]から[0,0,a,tar,a,a,0,...]と1要素後ろにずらした 

Update on 2016/02/02
array型ベースに書換え
'''

import numpy as np

def smoothing(signal, smoothing_range):
    """
    @param signal: 信号列
    @param smoothing_range: スムージングのwindow幅 
    
    偶数の場合は、該当する点より前の要素数が多くなる
    
    @return: 平滑化後のsignal値　array型
    """
    # window幅の半値を計算
    win = smoothing_range/2
    # 信号長の0配列を設定
    data = np.zeros((len(signal),), dtype=np.float32)
    # 平滑化
    for i in range(win, len(signal)-win):
        data[i] = np.average(signal[i-win+1:i+win+1])
#    # window幅の半値を計算
#    win = smoothing_range/2 
#    data = [0 for _ in range(win)]
#    # 平滑化
#    for i in range(win, len(signal)-win):
#        data.append(np.average(signal[i-win:i+win]))
#    data.extend([0 for _ in range(win)])

    return data


if __name__=="__main__":
    sig = [1,2,3,4,5,6,7,8,9,10]
    win=4
    res = smoothing(sig, win)
    print len(res), res