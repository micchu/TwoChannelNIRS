#coding:utf-8

'''
Created on 2016/01/27
@author: misato

動作確認済
'''

def slice_signal(signal, start_lag, step):
    """
    何個かおきに値をピックアップする（スライス）
    @param signal: 信号列 一次元array型
    @param start_lag: 初期のラグ
    @param step: 何個ずつ飛ばすか   
    """
    sliced = signal[start_lag:len(signal)-start_lag:step]
    return sliced

if __name__=="__main__":
    a = range(100)
    print slice_signal(a, 0, 10)
