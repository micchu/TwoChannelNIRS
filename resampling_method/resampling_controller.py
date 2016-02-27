#coding:utf-8

'''
Created on 2016/02/22

@author: misato
'''

import numpy as np

from resampling_method import bootstrap, smote

def resampling(label_list, feature_list, param):
    """
    リサンプリングの実行
    @param label_list: ラベル
    @param feature_list: 特徴量 <2次元array n sample * m features>
    @param param: パラメータモジュール  
    @return: 元のラベル+新しく作成されたラベルのリスト <n+x sample>
             元の特徴量+新しく作成された特徴量array <n+x sample * m feature>
    """
    if not hasattr(param, "RESAMPLING_TYPE"):
        raise NameError("resampling type not exists.")
        quit()
    
    if param.RESAMPLING_METHOD == "none": # リサンプリングしない
        return label_list, np.asarray(feature_list)
    # ラベルの一意リスト    
    distinct_label_list = set(label_list)
    
    new_label_list = []
    new_feature_list = []
    for label in distinct_label_list:
        # 各ラベルのデータを抽出
        extracted_feature_list = []
        for i in range(len(label_list)):
            if label == label_list[i]:
                extracted_feature_list.append(feature_list[i])
        
        # ダウンサンプリングについて検証
        if len(extracted_feature_list) >= param.RESAMPLING_SIZE:
            # ランダムに選択し、個数を減らす
            index_list = range(len(extracted_feature_list))
            param.RAND.shuffle(index_list)
            new_feature_list.extend([extracted_feature_list[j] for j in index_list[:param.RESAMPLING_SIZE]])
            new_label_list.extend([label for _ in range(param.RESAMPLING_SIZE)])
        # オーバーサンプリング時における各リサンプリング手法の実行
        else:
            if param.RESAMPLING_TYPE == "static": # 特定数までサンプルを増やす
                resampled_feature_list = None
                if param.RESAMPLING_METHOD == "bootstrap": 
                    # リサンプリングの実行（今、重複無しになっている）
                    resampled_feature_list = bootstrap.bootstrap(extracted_feature_list, param.RESAMPLING_SIZE - len(extracted_feature_list), param.RESAMPLING_N_SIZE, param.RAND, duplication = False)
                elif param.RESAMPLING_METHOD == "smote":
                    # リサンプリングの実行（今、重複無しになっている）
                    resampled_feature_list = smote.smote(extracted_feature_list, param.RESAMPLING_SIZE - len(extracted_feature_list), param.RESAMPLING_N_SIZE)
                # ラベルと特徴量の格納
                new_feature_list.extend(extracted_feature_list)
                new_feature_list.extend(resampled_feature_list)
                new_label_list.extend([label for _ in range(param.RESAMPLING_SIZE)])
                
            if param.RESAMPLING_TYPE == "replace": # 同サンプル数で置き換える
                resampled_feature_list = None
                if param.RESAMPLING_METHOD == "bootstrap": 
                    # リサンプリングの実行
                    resampled_feature_list = bootstrap.bootstrap(extracted_feature_list, len(extracted_feature_list), param.RESAMPLING_N_SIZE, param.RAND, duplication = False)
                elif param.RESAMPLING_METHOD == "smote":
                    # リサンプリングの実行
                    resampled_feature_list = smote.smote(extracted_feature_list, len(extracted_feature_list), param.RESAMPLING_N_SIZE)
                # ラベルと特徴量の格納
                new_feature_list.extend(resampled_feature_list)
                new_label_list.extend([label for _ in range(len(extracted_feature_list))])
    
    # array型に変換
    new_feature_array = np.asarray(new_feature_list)

    return new_label_list, new_feature_array
