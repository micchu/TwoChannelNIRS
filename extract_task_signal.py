#coding:utf-8

'''
Created on 2016/01/27
@author: misato

挙動確認済

Update on 2016/02/02
array型に対応させた
'''

import numpy as np

def extract_task_signal(signal, task_start, task_duration):
    """
    タスク期間中のデータを抽出する
    @param signal: 元のシグナルデータ array型
    @param task_start: タスクの開始タイミングのリスト
    @param task_duration:　タスクの継続期間
    @return: signal_array サンプル行×特徴次元列のarray型
    
    開始は0.0秒であることに注意
    1.0秒目のデータにアクセスするにはdata[9]ではなく、data[10]でアクセスする
    """
    # 最初の部分を切り出し
    signal_array = signal[task_start[0]:task_start[0]+task_duration]
    
    # vstackで積み上げて行く
    for i in range(1, len(task_start)):
        extracted_signal = signal[task_start[i]:task_start[i]+task_duration]
        signal_array = np.vstack((signal_array, extracted_signal)) 
        
    return signal_array
        
    