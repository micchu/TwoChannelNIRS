#coding:utf-8

'''
Created on 2016/01/27

@author: misato
'''
import traceback
import os, csv, datetime, random
import numpy as np

from my_error import MyError
import myParameter

from import_datafiles.label import import_label
from import_datafiles.signal import import_signal

import preprocessing_method.smoothing as prsm
import relabel as rela
import extract_task_signal
from feature_extraction.norm_zero import normalize_0point
from feature_extraction.slice_signal import slice_signal

import resampling_method.bootstarp as rebo
import resampling_method.smote as resm

from my_classifier.separate_dataset import separate_dataset_for_K_FOLD
import my_classifier.neural_network as mcnn
from my_classifier.proc_res import confusion_matrix as confmat
from my_classifier.proc_res import evaluation_regression as evre

PRINT_FLAG = True

def repeat_main():
    repeat_num = 1
    # none
    for seed in range(repeat_num):
        # パラメータモジュール
        param = myParameter.myParameter()
        # 乱数モジュール
        param.RAND = random.Random()
#        param.N_FOLD = 100
        # 実行
        main(param)
    
#    # replace
#    for seed in range(repeat_num):
#        # パラメータモジュール
#        param = myParameter.myParameter()
#        param.RESAMPLING_TYPE = "replace"
#        param.N_FOLD = 100
#        # 乱数モジュール
#        param.RAND = random.Random()
#        # 実行
#        main(param)


def main(param):    
    # 書込み対象のファイルを用意
    result_filename = "analysis/result/result.csv"
    if os.path.exists(result_filename):
        _directory, _filename = os.path.split(result_filename)
        _filename, _extention = os.path.splitext(_filename)
        now = datetime.datetime.now()
        _filename += "_" + now.strftime("%Y%m%d%H%M")
        result_filename = os.path.join(_directory, _filename + _extention)

    # 被験者IDリスト
    subject_id_list = ["amurakami", "ckinoshita", "hkimpara", "htanaka", "hwada", "kfukunaga", 
#                      "khayashinuma", "nishida", "robana", "sarita", "skatsurada", 
                      "kharada", "khayashinuma", "nishida", "robana", "sarita", "skatsurada", 
                      "syokoyama", "tishihara", "ttamaki", "ykohri", "yokada", "ysakaguchi"]
    subject_id_list = ["kharada", "hwada"]

    # 被験者
    for subject_id in subject_id_list:
        try:
            processing(subject_id, param, result_filename)
        except MyError as e:
            fp = open("analysis/log.txt", 'a')
            fp.write(subject_id + "\n")
            fp.write(e.value + "\n")
        except Exception, e:
            print "in"
            print traceback.format_exc()         
               
def processing(subject_id, param, result_filename):
    """
    被験者1人の識別
    """    
    if PRINT_FLAG:
        print "subject:", subject_id
    # fNIRSデータのディレクトリ
    signalfile_directory = "dataset/signal_data"
    # ラベルデータのディレクトリ
    labelfile_directory = "dataset/questionnaire_result"
    # 呈示ログのディレクトリ
#    logfile_directory = "/Users/misato/Documents/Research/Experiment/TwoChannelNIRS/presentation_log"
    
    if PRINT_FLAG:
        print "data loading..."

    ########################################################    
    # 前処理・特徴量抽出
    ########################################################    

    # list型 ラベルデータの読み込み（セッション別）
    label_file = os.path.join(labelfile_directory, subject_id + ".csv")
    label_list_per_session = import_label(label_file)
    
    # fNIRSデータとPresentationログの読み込み
    signal_data_per_session = []
    for s in range(param.SESSION_NUM):
        # matrix型 fNIRSデータの読み込み
        signal_file = os.path.join(signalfile_directory, subject_id, subject_id + str(s+1) + ".csv")
        signal_data_per_session.append(import_signal(signal_file)) # 中身はarray型
        # Presentationログの読み込みは未実装
        
    # Reラベル（ラベルの着け直し）
    if param.RELABEL_METHOD == "step":
        new_label_list_per_session = []
        for s in range(param.SESSION_NUM):
            new_label_list_per_session.append(rela.relabel_step(label_list_per_session[s], param.CLASS_NUM))
        label_list_per_session = new_label_list_per_session
    
    if PRINT_FLAG:
        print "preprocessing..."
        
    all_label_list = []
    all_feature_list = []
    # セッション毎に処理
    for s in range(param.SESSION_NUM):
        # 使用する特徴量の選択
        if param.TARGET_FEATURE == "totalHB":
            # 左右のtotalHBのデータのみを抽出 （元データに干渉しないよう、arrayを新しく生成）
            signal1 = np.array(signal_data_per_session[s][:,1].T) # 1次元array型
            signal2 = np.array(signal_data_per_session[s][:,4].T) # 1次元array型
       
            # 各データの単独処理
            preprocessed_signal_array1 = preprocessing(signal1, param)
            preprocessed_signal_array2 = preprocessing(signal2, param)
    
            # 左右データの結合 （列方向での結合）
            preprocessed_signal_array = np.c_[preprocessed_signal_array1, preprocessed_signal_array2]
                        
        # 全セッションデータの融合
        all_label_list.extend(label_list_per_session[s])
        all_feature_list.append(preprocessed_signal_array)
        
    # 行方向にsingalのarrayを連結
    all_feature_array = np.vstack(all_feature_list)
    # 特徴量の次元数を定数化
    feature_dimension_size = len(all_feature_array[0])

    if False:
        # 使用する特徴量の一次出力
        cc = csv.writer(open("tmp.csv", 'wb'))
        ret_array = np.reshape(np.array(all_label_list), (len(all_label_list),1))
        ret_array = np.c_[ret_array, all_feature_array]
        ret_list = ret_array.tolist()
        cc.writerows(ret_list)

    ########################################################    
    # 機械学習
    ########################################################    

    # K-foldクロスバリデーション用にデータセットを分割
    if param.N_FOLD == None: # N_FOLD定数が未定の場合、Leave-one-outとする
        param.N_FOLD = len(all_label_list) 
    if PRINT_FLAG:
        print "validation...", str(param.N_FOLD)+"-FOLD"
    dataset_label_list, dataset_feature_array = separate_dataset_for_K_FOLD(all_label_list, all_feature_array, param.N_FOLD, param.RAND)

    # K-foldクロスバリデーションに基づく識別率の計算
    default_accuracy = 0.0
    default_loss = 0.0
    all_test_label_list = []
    all_result_label_list = []
    all_result_loss_list = []
    for k in range(param.N_FOLD):
        if PRINT_FLAG:
            print "Fold: ", k
        # トレーニングデータとテストデータの分離
        # テストデータ
        test_label_list = dataset_label_list[k]
        test_feature_array = dataset_feature_array[k,]

        # トレーニングデータ
        training_label_list = []
        training_feature_list = []
        for j in range(param.N_FOLD):
            if k != j:
                training_label_list.extend(dataset_label_list[j])
                training_feature_list.append(dataset_feature_array[j,])
        training_feature_array = np.vstack(training_feature_list)
            
        # リサンプリング処理
        print "resampling..."
        resampled_training_label_list, resampled_training_feature_array = resampling(training_label_list, 
                                                                                     training_feature_array, param)
        # array型への変換
        resampled_training_feature_array.astype(np.float32)# = np.astype(resampled_training_feature_array, dtype=)
        test_feature_array.astype(np.float32)# = np.array(test_feature_list, dtype=)
        resampled_training_label_array = np.array(resampled_training_label_list, dtype=np.int32)
        test_label_array = np.array(test_label_list, dtype=np.int32)

        # トレーニングデータの並び替え
        mixed = resampled_training_label_array.reshape((len(resampled_training_label_array),1))
        mixed = np.c_[mixed, resampled_training_feature_array]            
        np.random.shuffle(mixed)
        resampled_training_label_array = mixed[:,0].flatten().astype(np.int32)
        resampled_training_feature_array = mixed[:,1:].astype(np.float32)

        # 識別器の構築
        print "train..."+param.LEARNING_METHOD+","+param.LEARNING_TYPE
        # 学習器の選択
        if param.LEARNING_METHOD == "NN":
            print "NN"
            # クラス分類型
            if param.LEARNING_TYPE == "class":
                print "class"
                # モデル生成
                nnobj = mcnn.network_model.set_model(feature_dimension_size, param, param.CLASS_NUM)
                # トレーニング
                mcnn.train.train_nn_class(nnobj, 
                                    resampled_training_feature_array, 
                                    resampled_training_label_array,
                                    param.BATCH_SIZE, param.EPOCH,
                                    print_flag=True,
                                    x_test = test_feature_array, 
                                    y_test = test_label_array,
                                    test_flag=True
                                    )
                # 識別器のテスト
                print "test..."
                acc, loss, result_label_list, result_loss_list, result_class_power_list = mcnn.test.test_nn_class(nnobj, test_feature_array, test_label_array)                
                # 識別結果の格納
                default_accuracy += acc
                default_loss += loss
                all_test_label_list.extend(test_label_list)
                all_result_label_list.extend(result_label_list)
                all_result_loss_list.extend(result_loss_list)
            # 回帰型
            elif param.LEARNING_TYPE == "regression":
                # モデル生成（出力層のユニット数は1で固定）
                nnobj = mcnn.network_model.set_model(feature_dimension_size, param, 1)
                # トレーニング
                mcnn.train.train_nn_regression(nnobj, 
                                        resampled_training_feature_array, 
                                        resampled_training_label_array,
                                        param.BATCH_SIZE, param.EPOCH,
                                        print_flag=True,
                                        x_test = test_feature_array, 
                                        y_test = test_label_array,
                                        test_flag=True
                                        )
                # 識別器のテスト
                print "test..."
                predicted_value_list, loss_list = mcnn.test.test_nn_regression(nnobj, test_feature_array, test_label_array)
                # 予測結果の格納
                default_loss += np.average(loss_list)
                all_test_label_list.extend(test_label_list)
                all_result_label_list.extend(predicted_value_list)
                all_result_loss_list.extend(loss_list)
        # N-FOLD終わり
        
    # 全体識別率
    default_accuracy /= param.N_FOLD
    default_loss /= param.N_FOLD
    
    if PRINT_FLAG:
        print 
        if param.LEARNING_TYPE == "class":
            print "acc:", default_accuracy, ", loss:", default_loss
        elif param.LEARNING_TYPE == "regression":
            print "loss:", default_loss
    ################## 識別計算ここまで
    
    # 結果の格納
    if PRINT_FLAG:
        print "logging..."
        
    # ラベルリストの作成
    label_list = range(param.CLASS_NUM)

    if param.LEARNING_TYPE == "class":
        # 混合行列の生成
        confusion_mat = confmat.get_confusion_matrix(label_list, all_test_label_list, all_result_label_list)
        # クラス毎の適合率、再現率、F-measureの計算
        precision, recall, f_measure, result_mat = confmat.calculate_accuracy_per_label(label_list, confusion_mat)        
        # 相関値の計算
        corr, p_value = evre.get_correlation(all_test_label_list, all_result_label_list)
        # 距離の計算
        dist = evre.get_distance(all_test_label_list, all_result_label_list)
        # 結果の書込み
        logging_result_class(subject_id, result_filename, default_accuracy, default_loss, precision, recall, f_measure, dist, corr, p_value, result_mat, all_test_label_list, all_result_label_list, all_result_loss_list)
    
    elif param.LEARNING_TYPE == "regression":
        # 相関値の計算
        corr, p_value = evre.get_correlation(all_test_label_list, all_result_label_list)
        # クラス毎のロス平均の計算（クラス毎の相関はあまり意味ないので非計算）
        loss_per_class = evre.get_loss_per_class(label_list, all_test_label_list, all_result_loss_list)                                
        # 距離の計算
        dist = evre.get_distance(all_test_label_list, all_result_label_list)
        # 結果の書込み
        logging_result_regression(subject_id, result_filename, dist, corr, p_value, default_loss, loss_per_class, 
                           all_test_label_list, all_result_label_list, all_result_loss_list)
    # END    

    
def resampling(label_list, feature_list, param):
    """
    リサンプリングの実行
    @param label_list: ラベル
    @param feature_list: 特徴量 二次元array型(サンプル数×特徴次元数)
    @param param: パラメータモジュール  
    @return: 元のラベル+新しく作成されたラベルのリスト、元の特徴量+新しく作成された特徴量リスト
    """
    if param.RESAMPLING_TYPE == "none": # リサンプリングしない
        return label_list, feature_list

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
                
        # 各リサンプリング手法の実行
        if param.RESAMPLING_TYPE == "static": # 特定数までサンプルを増やす
            resampled_feature_list = None
            if param.RESAMPLING_METHOD == "bootstrap": 
                # リサンプリングの実行（今、重複無しになっている）
                resampled_feature_list = rebo.bootstrap(extracted_feature_list, param.RESAMPLING_SIZE - len(extracted_feature_list), param.RESAMPLING_N_SIZE, param.RAND, duplication = False)
            elif param.RESAMPLING_METHOD == "smote":
                # リサンプリングの実行（今、重複無しになっている）
                resampled_feature_list = resm.smote(extracted_feature_list, param.RESAMPLING_SIZE - len(extracted_feature_list), param.RESAMPLING_N_SIZE)
            # ラベルと特徴量の格納
            new_feature_list.extend(extracted_feature_list)
            new_feature_list.extend(resampled_feature_list)
            new_label_list.extend([label for _ in range(param.RESAMPLING_SIZE)])
            
        if param.RESAMPLING_TYPE == "replace": # 同サンプル数で置き換える
            resampled_feature_list = None
            if param.RESAMPLING_METHOD == "bootstrap": 
                # リサンプリングの実行
                resampled_feature_list = rebo.bootstrap(extracted_feature_list, len(extracted_feature_list), param.RESAMPLING_N_SIZE, param.RAND, duplication = False)
            elif param.RESAMPLING_METHOD == "smote":
                # リサンプリングの実行
                resampled_feature_list = resm.smote(extracted_feature_list, len(extracted_feature_list), param.RESAMPLING_N_SIZE)
            # ラベルと特徴量の格納
            new_feature_list.extend(resampled_feature_list)
            new_label_list.extend([label for _ in range(len(extracted_feature_list))])
    
    # array型に変換
    new_feature_array = np.asarray(new_feature_list)

    return new_label_list, new_feature_array
    

def preprocessing(signal, param):
    """
    前処理における各シグナルの単独処理（確認済）
    @param singal: fNIRS信号 1次元array型
    @param param: パラメータモジュール  
    
    @return: 前処理終了後のシグナル値 2次元array型（サンプル数×特徴次元数）
    """            
    # データの前処理
    # HPF
    # pass
    
    # 平滑化
    smoothed_signal = prsm.smoothing(signal, param.SMOOTHING_LENGTH)
    
    # タスク期間中のデータの切り出し
    signal_array = extract_task_signal.extract_task_signal(smoothed_signal, param.TASK_START, param.TASK_DURATION)
    
    # 特徴量への変換
    # 1秒おきの抽出
    sliced_singal_list = []
    for i in range(len(signal_array)):
        new_sig = slice_signal(signal_array[i,], param.SAMPLING_START_LAG, param.SAMPLING_RATE)
#        print type(new_sig), new_sig
        #print "slice", new_sig
        new_sig = normalize_0point(new_sig)
        #print "nor", new_sig
        sliced_singal_list.append(new_sig)

    sliced_singal_array = np.asarray(sliced_singal_list)

    return sliced_singal_array

def logging_result_class(subject_id, target_filename, accuracy, loss, precision, recall, 
                         f_measure, dist, corr, p_value, result_mat,
                         all_test_label_list, all_result_label_list, all_result_loss_list):
    """
    クラス分類用の結果書き出し
    @param subject_id: 被験者ID 
    @param target_filename: 結果の出力先 
    @param accuracy: 識別率
    @param loss: 誤差 
    @param precision: クラス平均適合率  
    @param recall: クラス平均再現率
    @param f_measure: クラス平均F-Measure
    @param dist: 距離平均 
    @param corr:相関係数
    @param p_value: 相関係数のp値  
    @param result_mat: 各クラスの結果
    @param all_test_label_list: 全識別個体のラベルリスト
    @param all_result_label_list: 全識別個体の識別・推定結果のリスト
    @param all_result_loss_list: 全識別個体のロスのリスト
    """
    # 書込み対象のファイルを開く
    cw = csv.writer(open(target_filename, 'ab'))
    
    # 書込みデータの整形
    res_row = [subject_id, accuracy, loss, precision, recall, f_measure, dist, corr, p_value]
    # 各クラスの結果は、クラス毎に繰り返し
    res_row.extend(list(result_mat.flatten()))
    # 全ラベル・識別率・ロスの追加
    res_row.append("#")
    res_row.extend(all_test_label_list)
    res_row.append("#")
    res_row.extend(all_result_label_list)
    res_row.append("#")
    res_row.extend(all_result_loss_list)
    
    # 書込み
    cw.writerow(res_row)

def logging_result_regression(subject_id, target_filename, dist, corr, p_value, loss, loss_per_class, 
                              all_test_label_list, all_result_label_list, all_result_loss_list):
    """
    回帰の場合の結果書き出し
    @param subject_id: 被験者ID 
    @param target_filename: 結果の出力先 
    @param loss: 全誤差の平均
    @param dist: 距離平均 
    @param corr: スピアマンの相関係数 （非正規）
    @param p_value: 相関係数のp値 
    @param loss_per_class: クラス別のロス平均
    @param all_test_label_list: 全識別個体のラベルリスト
    @param all_result_label_list: 全識別個体の識別・推定結果のリスト
    @param all_result_loss_list: 全識別個体のロスのリスト
    """
    # 書込み対象のファイルを開く
    cw = csv.writer(open(target_filename, 'ab'))
    
    # 書込みデータの整形
    res_row = [subject_id, loss, dist, corr, p_value ]
    # 各クラスの結果は、クラス毎に繰り返し
    res_row.extend(loss_per_class)
    # 全ラベル・識別率・ロスの追加
    res_row.append("#")
    res_row.extend(all_test_label_list)
    res_row.append("#")
    res_row.extend(all_result_label_list)
    res_row.append("#")
    res_row.extend(all_result_loss_list)

    # 書込み
    cw.writerow(res_row)
    
    
if __name__=="__main__":
    repeat_main()