#coding:utf-8

'''
Created on 2016/01/27

@author: misato
'''

class myParameter():
    # 全セッション数
    SESSION_NUM = 4
    
    ########################################################    
    # ラベル・クラスの扱いに関するパラメータ
    ########################################################    
    # Relabelパラメータ
    # stepは固定、varは分散最小化
    # noneはリラベル無し(正確には/10をしておく)
    RELABEL_METHOD = "step" 
    # クラスラベルのマッピング{元のラベル名，割り当てラベル}
    CLASS_NUM = 5
    
    ########################################################    
    # 前処理・特徴量抽出に関するパラメータ
    ########################################################    
    # 使用する特徴量
    # "totalHB"左右の脳血流
    # "totalHB1"1cm深度の左右の脳血流
    # "totalHB3"3cm深度の左右の脳血流
    #時刻    脳血流(左)    脳血流(左1cm)    脳血流(左3cm)    脳血流(右)    脳血流(右1cm)    脳血流(右3cm)    脈拍(左)    脈拍(右)    LF/HF(左)    LF/HF(右)    温度    X軸角度    Y軸角度    Z軸角度    深呼吸デキ度    チャートX軸    チャートY軸    チャート象限    チャート円半径    マーキング(1〜32)                                                                                                                            
    TARGET_FEATURE = "totalHB"
    
    # タスク期間を指定するパラメータ
    # 最初のレストが15秒
    TASK_START = [i*250 + 150 for i in range(25)]
    TASK_DURATION = 100 # 0.1[s]単位、今回は10秒間
    
    # 特徴量の種類
    # "raw" (要SMOOTHING_LENGTH)
    # "diff" (要SMOOTHING_LENGTH)
    # "wavelet"
    FEATURE_TYPE = "raw" # そのまま
    # 平滑化
    SMOOTHING_LENGTH = 50 # 0.1s単位、0の場合はSmoothingしない
    # HPFをするか
    HPF_FLAG = False
    # HPFの帯域(Hz)
    HPF_BAND = None
    # 再サンプリングレート（平滑化等終わった後）
    SAMPLING_RATE = 10 # 0.1s単位
    SAMPLING_START_LAG = 0 # 0秒目からとる
    # SSとSMの計算範囲
    SSSM_START = 20 # 2秒目から
    SSSM_WINDOW = 50 # 7秒目までがデフォルト値
    
    ########################################################    
    # リサンプリング関連のパラメータ
    ########################################################    
    # リサンプリングタイミング
    RESAMPLING_TARGET = "training" # "all" or "training"
    # リサンプリングタイプ
    # "none"しない
    # "static"ある個数まで追加 (要RESAMPLING_SIZE)
    # "double"倍に増やす
    # "replace_static"ある個数まで全サンプル置き換え (要RESAMPLING_SIZE)
    # "replace"全サンプルを個数をそのままに置き換え
    RESAMPLING_TYPE = "none" 
    RESAMPLING_SIZE = 40
    # リサンプリング手法
    # "bootstrap", "SMOTE"
    RESAMPLING_METHOD = "bootstrap"
    # 各手法で用いる個数
    RESAMPLING_N_SIZE = 5     

    ########################################################    
    # 機械学習のパラメータ
    ########################################################    
    # K-fold crossvalidation
    # None: Leave one outとなる
    N_FOLD = 10
    
    # 学習器の選択
    # "NN" ニューラルネットワーク
    # "LDA" 線形識別
    # "SVM" サポートベクターマシン
    LEARNING_METHOD = "NN"
    
    # クラス分類か回帰か？
    # "class" クラス分類，"regression" 回帰
    LEARNING_TYPE = "class"
    
    # NN用パラメータ 
    BATCH_SIZE = 1 # 1データずつ重みを更新
    EPOCH = 100 #200回繰り返す
    ACTIVATION_FUNCTION = "softmax"
    NODE_NUM_1LAYER = 30 #中間層の数
    
    # 乱数モジュールへのリファレンス
    RAND = None

    if True: # Check用
        N_FOLD = 10
        EPOCH = 50 # 200回繰り返す
        BATCH_SIZE = 5 # 5データずつ重みを更新
        SMOOTHING_LENGTH = 30
        

