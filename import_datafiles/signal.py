#coding:utf-8

'''
Created on 2016/01/27
@author: misato

動作確認済

Update on 2016/1/29
HOT121Bのログ形式から処理しやすい形式へ変換
合わせてデータの読込み方を変更（csvモジュール→numpyモジュール）
'''

import csv
import numpy as np

def import_signal(target_file):
    """
    ファイルからデータ行を読込む
    @param target_file: 読込み対象の絶対パス
    @return: 読込んだデータのarray型行列データ（行：時間、列：データ種類）
    
    動作確認済
    """
    data = np.loadtxt(target_file, delimiter=",", skiprows=1, dtype=np.float32)
            
    return data

def convert_signaldata(target_file, new_target_file):
    """
    HOT121Bのログファイルをまともなファイル形式に修正
    @param target_file: 読込み対象の絶対パス
    @param new_target_file: 出力先の絶対パス 
    @return: なし
    """
    cr = csv.reader(open(target_file, 'r'))
    cw = csv.writer(open(new_target_file, 'w'))
    data_flag = False
    new_header = ["Time", 
                  "TotalHb_Left", "TotalHb_Left_1cm", "TotalHb_Left_3cm", 
                  "TotalHb_Right", "TotalHb_Right_1cm", "TotalHb_Right_3cm", 
                  "HR_Left", "HR_Right", 
                  "LF/HF_Left", "LF/HF_Right",
                  "TemperatureC", 
                  "Slope_X", "Slope_Y", "Slope_Z", 
                  "DeepBreath", 
                  "Chart_X", "Chart_Y", "Chart_Quadrant", "Chart_Radius", 
                  "Mark"]
    
    cw.writerow(new_header)
    for r in cr:
        if len(r) > 30: # データ行は列数が22以上になっているので
            cw.writerow([float(a) for a in r[:21]])
            
            
if __name__=="__main__":
    # import_signalメソッドの動作確認用 
    if True:
        target_file = "/Users/misato/Documents/Research/Experiment/TwoChannelNIRS/signal_data/amurakami/amurakami1.csv"
        print import_signal(target_file)
    
    # signalデータファイルのコンバート
    if False:
        subject_id_list = ["amurakami", "ckinoshita", "hkimpara", "htanaka", "hwada", "kfukunaga", 
                          "kharada", "khayashinuma"]
        subject_id_list = ["nishida", "robana", "sarita", "skatsurada", 
                          "syokoyama", "tishihara", "ttamaki", "ykohri", "yokada", "ysakaguchi"]
        subject_id_list = ["tsato"]
        subject_id_list = ["jnishida"]
        for subject_id in subject_id_list:
            for s in [1,2,3,4]:
                target_file = "/Users/misato/Documents/Research/Experiment/TwoChannelNIRS/_backup_signal_data/"+subject_id+"/"+subject_id+str(s)+".csv"
                new_target_file = "/Users/misato/Documents/Research/Experiment/TwoChannelNIRS/signal_data/"+subject_id+"/"+subject_id+str(s)+".csv"
                convert_signaldata(target_file, new_target_file)

