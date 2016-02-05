#coding:utf-8

'''
Created on 2016/01/30
@author: misato

Update on 2016/02/01 
初期点(要素0)の設定をミスってたので修正（for文回している最中に0になるのに、それを参照して引算してた<汗）

Update on 2016/02/02
array型に対応
'''

def normalize_0point(signal, standard = 0):
    """
    0点補正
    @param singal: 信号 一次元array型
    @keyword standard: 基準点のINDEX（defaultは0で、最初の要素が0点に補正される）  
    """
    first_point = signal[0]
    signal -=  first_point
    return signal


if __name__=="__main__":
    print normalize_0point([1,2,3,4,5])
    
