#coding:utf-8

'''
Created on 2016/02/16
@author: misato
'''


import csv
import numpy as np

def import_log(target_file, param):
    """
    トライアルの開始時間を0.1[s]単位で抽出
    @param target_file: 対象となるPresentationログファイル 
    @param param: パラメータモジュール 
    @return: シナリオ開始から各タスクの開始時間のリスト 0.1[s]単位
    動作確認済
    """
    if param.FS != 10.0:
        raise NameError("This script can processing 10.0 Hz data.")
    # データの読み出しとヘッダ行を飛ばす
    cr = csv.reader(open(target_file, 'Ur'), delimiter="\t")
    for i in range(5):
        cr.next()
    # データ取り出し
    trial_start = []
    for r in cr:
        if len(r) > 0:
            _time = int(r[3])
            trial_start.append(int(round(_time/1000.0, 0)))
        else:
            break;
    
    return trial_start

if __name__=="__main__":
    import myParameter
    param = myParameter.myParameter()
    target_file = "/Users/misato/Documents/eclipse/workspace/TwoChannelNIRS/dataset/presentation_log/ysakaguchi/session1.log"
    print import_log(target_file, param)
            