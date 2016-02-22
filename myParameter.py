#coding:utf-8

'''
Created on 2016/01/27

@author: misato
'''

class myParameter():
    # 今回使用するセッション ※ 特に実装にはまだ組み込んでいない
    USED_SESSION_LIST = [1,2,3,4]
#    USED_SESSION_LIST = [1]
    # 全セッション数
    SESSION_NUM = len(USED_SESSION_LIST)
    # NIRSの計測周波数
    FS = 10.0 
    # 総サンプル数
    TOTAL_SAMPLE_NUM = int(640*FS)
    
    ########################################################    
    # ラベル・クラスの扱いに関するパラメータ
    ########################################################    
    # Relabelパラメータ
    # 今回使用するラベル ※ 特に実装にはまだ組み込んでいない
    USED_LABEL_LIST = [1,2,3,4,5,6,7,8,9,10]
    # クラスラベル
    CLASS_LIST = [0,1,2,3,4]
    CLASS_NUM = 5
    # noneはリラベル無し、varは分散最小化, mappigはRELABEL_MAPPINGを使用する
    # scalingはregression用で*0.1-0.5する
    RELABEL_METHOD = "scaling"
    RELABEL_MAPPING = {1:0,2:0,
                       3:1,4:1,
                       5:2,6:2,
                       7:3,8:3,
                       9:4,10:4}
    
    ########################################################    
    # 前処理・特徴量抽出に関するパラメータ
    ########################################################    
    # 使用する特徴量
    # "HBX"左右の脳血流
    # "HBX1"1cm深度の左右の脳血流
    # "HBX3"3cm深度の左右の脳血流
    #時刻    脳血流(左)    脳血流(左1cm)    脳血流(左3cm)    脳血流(右)    脳血流(右1cm)    脳血流(右3cm)    脈拍(左)    脈拍(右)    LF/HF(左)    LF/HF(右)    温度    X軸角度    Y軸角度    Z軸角度    深呼吸デキ度    チャートX軸    チャートY軸    チャート象限    チャート円半径    マーキング(1〜32)                                                                                                                            
    TARGET_SIGNAL = "HBX"
    
    # タスク期間を指定するパラメータ
    # 最初のレストが15秒
    TASK_START = [i*250 + 150 for i in range(25)]
    TASK_DURATION = 150 # 0.1[s]単位、今回は10秒間
    
    # 前処理の掛け方
    # "butter": バターワースバンドパスフィルタ
    # "none": フィルタリング無し
    FILTER_TYPE = "butter"
    # "smoothing": 平滑化のオンオフ
    SMOOTHING_TYPE = True 
    # 平滑化
    SMOOTHING_LENGTH = 10 # 0.1s単位、0の場合はSmoothingしない
    # BPFの帯域(Hz) HPFとLPFのカットオフ周波数[Hz]
    #BPF_BAND = [0.03, 0.3] # 実験デザイン考えるとこっちな気もするんだけど...
    #BPF_BAND = [0.05, 0.3]
    BPF_BAND = [0.05, 0.3]
    BUTTER_WORTH_ORDER = 3 # ButterWorthフィルタの強度
    
    # 微分
    # 0: 微分しない、1,2: 1次、2次微分
    DIFF = 0
    
    # 特徴量の種類
    # "SLICE" 時系列信号の切り出し
    # "SSSM" SSとSMの計算
    FEATURE_TYPE = "SSSM"
    # 時系列型特徴量の切り出しパラメータ（平滑化等終わった後）
    SLICE_RATE = 10 # 0.1s単位
    SLICE_START_LAG = 0 # 0秒目からとる
    # SSとSMの計算範囲
    SSSM_START = 20 # 2秒目から
    SSSM_WINDOW = 50 # 7秒目までがデフォルト値
    
    # ノイズ処理
    # 10sample中に0.1以上の変動が発生した場合、ノイズとする
    NOISY_THRESHOLD = 0.1
    NOISY_T_RANGE = 10
    AFTER_NOISE = 30 # ノイズ発生後3秒間もまたノイズ影響領域とする
#    HPF_BAND = [0.5, None]
    
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
    HIDDEN_LAYER_NUM = 2 # 中間層の数 （変更においてはモデルとforward関数双方の修正が必要）
    NODE_NUM_1LAYER = 30 # 30 or 8
    NODE_NUM_2LAYER = 30 # 15 or 6
    NODE_NUM_3LAYER = 15 # 15 or 6
    BATCH_SIZE = 1 # 1データずつ重みを更新
    EPOCH = 100 #200回繰り返す
    ACTIVATION_FUNCTION = "softmax"
    
    # 乱数モジュールへのリファレンス
    RAND = None

    if True: # Check用
        N_FOLD = 10
        EPOCH = 100 # 100回繰り返す
        BATCH_SIZE = 1 # 5データずつ重みを更新

        TARGET_SIGNAL = "HBX3"
        BPF_BAND = [0.05, 0.5]
        LEARNING_TYPE = "class" #"regression"
        RESAMPLING_METHOD = "bootstrap" # "SMOTE"
        RESAMPLING_TYPE = "static" # "none" 
        RESAMPLING_SIZE = 30
        RELABEL_METHOD = "mapping"
        RELABEL_MAPPING = {1:0,2:0,
                           3:1,4:1,
                           5:2,6:2,
                           7:3,8:3,
                           9:4,10:4}
        # 上書き
        RELABEL_MAPPING = {1:0,2:0,
                           3:0,8:1,
                           9:1,10:1}
#        # さらに上書き
#        RELABEL_MAPPING = {1:0,2:0,
#                           3:0,5:1,
#                           6:1,7:1}
        CLASS_NUM = 2
        TASK_DURATION = 150 # 0.1[s]単位、今回は10秒間
        NOISY_THRESHOLD = 0.1 # ノイズ閾値
        NOISY_T_RANGE = 10 # ノイズ判定範囲
        if False:
            FEATURE_TYPE = "SSSM"
            NODE_NUM_1LAYER = 10 # 30 or 10
            NODE_NUM_2LAYER = 10 # 30 or 10
            SMOOTHING = False
        if True:
            FEATURE_TYPE = "SMSK"
            TASK_DURATION = 100 # 0.1[s]単位、今回は10秒間
            NODE_NUM_1LAYER = 10 # 30 or 10
            NODE_NUM_2LAYER = 10 # 30 or 10
            SMOOTHING = False
        if False:
            FEATURE_TYPE = "SLICE"
            NODE_NUM_1LAYER = 30 # 30 or 10
            NODE_NUM_2LAYER = 30 # 30 or 10
            SMOOTHING_TYPE = True  #平滑化のONOFF
        if RELABEL_METHOD == "mapping":
            # クラスラベル
            CLASS_LIST = [0,1]
            CLASS_NUM = 2
 
