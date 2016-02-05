#coding:utf-8

'''
Created on 2016/01/27
@author: misato

動作確認済

Update on 2016/1/29
アンケートのデータファイルの形式を整理（参照する列数を変更）
'''

import csv
import numpy as np

def import_label(target_file):
    """
    ファイルからラベルデータを読込み、数値化した上でセッション毎に分けて返す
    @param target_file: 読込み対象の絶対パス
    @return: 読込んだデータのリストデータ 行：セッション、列：呈示数
    """
    cr = csv.reader(open(target_file, 'Ur'))
    cr.next() # ヘッダ行を飛ばす

    session = 1
    label_in_sessions = []
    label_in_a_session = []
    for r in cr:
        if session != int(r[0]):
            session += 1
            label_in_sessions.append(label_in_a_session)
            label_in_a_session = []
        label_in_a_session.append(int(r[4]))
    label_in_sessions.append(label_in_a_session)

    return label_in_sessions
    
if __name__=="__main__":
    target_file = "/Users/misato/Documents/Research/Experiment/TwoChannelNIRS/questionnaire_result/amurakami.csv"
    print import_label(target_file)
