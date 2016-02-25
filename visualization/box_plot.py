#coding:utf-8

'''
Created on 2016/02/25

@author: misato
'''

import matplotlib.pyplot as plt

def box_plot(data_list, label_list, y_axis_name, chance_level, _title = ""):
    """
    箱ひげ図のプロット
    @param data_list: データのリスト <2次元list n labels * m samples>
    @param label_list: 横軸のラベル名のリスト  <1次元list n labels>
    @param y_axis_name:  縦軸の名前
    @param chance_level: チャンスレベル 
    @keyword _title: タイトル 
    """
    # 図
    fig = plt.figure()
    subfig = fig.add_subplot(111)
    subfig.set_title(_title)
    
    # データのセット
    subfig.boxplot(data_list)
    # 横軸のラベルのセット
    subfig.set_xticklabels(label_list, rotation=45, fontsize = 8)
    # 縦軸の設定
    plt.ylabel(y_axis_name)
    plt.ylim([0.0,1.0])
    # グリッド表示
    plt.grid()
    # チャンスレベル設定
#    plt.abline(a = [0, chance_level], b = [3, chance_level])
#    plt.abline(h=chance_level)
    
    # 表示
    plt.show()

if __name__=="__main__":
    data_list = [[0.1,0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5,0.6]]
    label_list = ["a", "b"]
    y_axis_name = "test"
    chance_level = 3
    
    box_plot(data_list, label_list, y_axis_name, chance_level, _title="a")
