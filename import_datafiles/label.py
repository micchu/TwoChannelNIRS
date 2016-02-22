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

def import_label(target_file, used_session_list):
    """
    ファイルからラベルデータを読込み、数値化した上でセッション毎に分けて返す
    @param target_file: 読込み対象の絶対パス
    @param used_session_list: 使用するセッション番号 
    @return: 読込んだデータのリストデータ 行：セッション、列：呈示数
    """
    cr = csv.reader(open(target_file, 'Ur'))
    cr.next() # ヘッダ行を飛ばす

    label_in_sessions = [[] for _ in range(len(used_session_list))]
#    label_in_a_session = []
    for r in cr:
        ses = int(r[0])
        if ses in used_session_list:
            label_in_sessions[used_session_list.index(ses)].append(int(r[4]))
    
    return label_in_sessions
    
if __name__=="__main__":
    target_file = "../dataset/questionnaire_result/amurakami.csv"
    ret = import_label(target_file, [2,4])
    print ret
    print len(ret)
