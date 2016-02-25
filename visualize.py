#coding:utf-8

'''
Created on 2016/02/25
@author: misato

結果の視覚化
'''

import os
import numpy as np
import pandas as pd

from visualization.con_csv import read_data, read_table, generate_pivot
from visualization.box_plot import box_plot

def visualize():
    # データ置き場
    target_directory = "analysis/result/Comparison_20160222"
    
    # 視覚化対象
    target_method = "LDA"
    target_data = "accuracy"
    plot_type = "box"
    
    # 描画
    parameter_list, filename_list = read_table(os.path.join(target_directory, target_method, "table.csv"))
    parametername_list = ["_".join(p) for p in parameter_list]
    data_list = []
    for i in range(len(parameter_list)):
        target_filename = os.path.join(target_directory, target_method, filename_list[i])
        subject, data = read_data(target_filename, target_data)
        data_list.append(data)
    
    box_plot(data_list[:36], parametername_list[:36], target_data, 0.0, _title = "HBX")
    box_plot(data_list[36:72], parametername_list[36:72], target_data, 0.0, _title = "HBX3")
    box_plot(data_list[72:], parametername_list[72:], target_data, 0.0, _title = "HBX1")
    

def convert_data():
    """
    統一ピボット形式への変換
    """
    # データ置き場
    target_directory = "analysis/result/Comparison_20160222"
    
    # 視覚化対象
    target_method = "ANN"
    #target_data = "accuracy"
    target_data = "all"
    plot_type = "box"
    
    # 描画
    parameter_list, filename_list = read_table(os.path.join(target_directory, target_method, "table.csv"))
    parametername_list = ["_".join(str(p)) for p in parameter_list]
    data_list = []
    for i in range(len(filename_list)):
        target_filename = os.path.join(target_directory, target_method, filename_list[i])
        _subject_list, _data_list = read_data(target_filename, target_data)
        data_list.append(_data_list)

    header_list = ["signal", "re_method", "re_size", "feature", "nfold", 
                   "subject_id","accuracy","loss","average_precition",
                   "average_recall","average_Fmeasure","average_distance","corr","p_value"]        
    output_filename = os.path.join(target_directory, target_method + ".csv")
    generate_pivot(header_list, parameter_list, data_list, output_filename)
    
#    parameter_array = np.array(parameter_list)
#    data_array = np.array(data_list)
#    marge_array = np.hstack([parameter_array, data_array.reshape(len(data_list),1)])
#    print marge_array

if __name__=="__main__":
    convert_data()
