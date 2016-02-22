#coding:utf-8

'''
Created on 2016/02/05

@author: misato
'''

import scipy.signal
from numpy import nonzero

from scipy.signal import butter, lfilter


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """
    バターワースフィルタによるバンドバス
    @param signal: 信号列 array型
    @param lowcut: HPFの帯域（低周波数側）[Hz]
    @param highcut: LPFの帯域（高周波数側） [Hz]
    @param fs: サンプリング周波数[Hz] 
    """
    def butter_bandpass(lowcut, highcut, fs, order=5):
        """
        @param lowcut: 帯域（低周波数側）
        @param highcut: 帯域（高周波数側） 
        @param fs: サンプリング周波数 
        """
        if highcut == None:
            # ハイパスフィルタ
            nyq = 0.5 * fs
            low = lowcut / nyq
            b, a = butter(order, low, btype='high')
            return b, a
        else:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
    
    # フィルタの形成
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # フィルタリング
    y = lfilter(b, a, signal)
    return y
