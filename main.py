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
from import_datafiles.presentation_log import import_log

import preprocessing_method.smoothing as prsm
import preprocessing_method.bandpassfilter as prba
import relabel as rela
import extract_task_signal
from feature_extraction.norm_zero import normalize_0point
from feature_extraction.slice_signal import slice_signal
from feature_extraction.SSSM import get_SS_SM
from feature_extraction.SMSK import get_SM_SK

from resampling_method import resampling_controller as recon

from my_classifier.separate_dataset import separate_dataset_for_K_FOLD
import my_classifier.neural_network as mcnn
import my_classifier.lda as mclda
import my_classifier.svm as mcsvm
from my_classifier.proc_res import confusion_matrix as confmat
from my_classifier.proc_res import evaluation_regression as evre
from matplotlib.pyplot import axis

PRINT_FLAG = True

def repeat_main():
    repeat_num = 1

    # パラメータモジュール

    for lm in ["SVM", "LDA", "ANN"]:
        for ts in ["HBX", "HBX3", "HBX1"]:
            for re in ["none"]:
#            for re in ["none", "bootstrap", "smote"]:
                for fe in ["SSSM", "SMSK", "SLICE"]:
                    
                    if re in ["bootstrap", "smote"]:
                        for rs in [20, 30, 40]:
                            for nf in [10, None]:
                                param = myParameter.myParameter()
                                # 乱数モジュール
                                param.RAND = random.Random()
                                param.LEARNING_METHOD = lm
                                param.TARGET_SIGNAL = ts
                                param.RESAMPLING_METHOD = re      
                                param.RESAMPLING_SIZE = rs
                                param.FEATURE_TYPE = fe
                                param.N_FOLD = nf
                                
                                fp = open("memo.txt", 'a')
                                fp.write(",".join([str(mm) for mm in [lm, ts, re, rs, fe, nf]]))
                                fp.write("\n")
                                fp.close()
                                
                                # 実行
                                _addfilename = lm+ts+re+str(rs)+fe+str(nf)
                                main(param, addfilename = _addfilename)
    
                    elif re == "none":  
                        rs = 0          
                        for nf in [10, None]:
                            param = myParameter.myParameter()
                            # 乱数モジュール
                            param.RAND = random.Random()
                            param.TARGET_SIGNAL = ts
                            param.RESAMPLING_METHOD = re        
                            param.RESAMPLING_SIZE = rs
                            param.FEATURE_TYPE = fe
                            param.N_FOLD = nf
                            
                            fp = open("memo.txt", 'a')
                            fp.write(",".join([str(mm) for mm in [lm, ts, re, rs, fe, nf]]))
                            fp.write("\n")
                            fp.close()
                            
                            # 実行
                            _addfilename = lm+ts+re+str(rs)+fe+str(nf)
                            main(param, addfilename = _addfilename)
    

def main(param, addfilename = None):    
    # 書込み対象のファイルを用意
    result_filename = "analysis/result/result.csv"
    if os.path.exists(result_filename):
        _directory, _filename = os.path.split(result_filename)
        _filename, _extention = os.path.splitext(_filename)
        now = datetime.datetime.now()
        if addfilename:
            _filename += "_" + now.strftime(addfilename+"_%Y%m%d%H%M")
        else:
            _filename += "_" + now.strftime("%Y%m%d%H%M")
        result_filename = os.path.join(_directory, _filename + _extention)

    # 被験者IDリスト
    subject_id_list = ["amurakami", "ckinoshita", "hkimpara", "htanaka", "hwada", "kfukunaga", 
                      "kharada", "khayashinuma", "nishida", "robana", "sarita", "skatsurada", 
                      "syokoyama", "tishihara", "ttamaki", "yokada", "ysakaguchi"]
    # selected
#    subject_id_list = ["ckinoshita"]

    # 被験者
    for subject_id in subject_id_list:
        try:
            processing(subject_id, param, result_filename)
        except MyError as e:
            fp = open("analysis/log.txt", 'a')
            fp.write(subject_id + "\n")
            fp.write(e.value + "\n")
            fp.write(str(traceback.format_exc()) + '\n')
            fp.close()
        except Exception, e:
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
    logfile_directory = "dataset/presentation_log"
    
    if PRINT_FLAG:
        print "data loading..."

    ########################################################    
    # 前処理・特徴量抽出
    ########################################################    
    #### 読み込み
    # list型 ラベルデータの読み込み（セッション別）
    label_file = os.path.join(labelfile_directory, subject_id + ".csv")
    label_list_per_session = import_label(label_file, param.USED_SESSION_LIST)
    
    # fNIRSデータとPresentationログの読み込み
    signal_data_per_session = []
    log_per_session = []
    for s in param.USED_SESSION_LIST:
        # matrix型 fNIRSデータの読み込み
        signal_file = os.path.join(signalfile_directory, subject_id, subject_id + str(s) + ".csv")
        signal_data_per_session.append(import_signal(signal_file)) # 中身はarray型
        # Presentationログの読み込み
        log_file =  os.path.join(logfile_directory, subject_id, "session" + str(s) + ".log")
        log_per_session.append(import_log(log_file, param))
        
    # Reラベル（ラベルの着け直し）
    label_list_per_session = relabeling(label_list_per_session, param)

    #### 前処理
    if PRINT_FLAG:
        print "preprocessing...Signal is %s, Filter is %s, Feature is %s" % (param.TARGET_SIGNAL, param.FILTER_TYPE, param.FEATURE_TYPE)

    all_label_list = []
    all_feature_list = []
    # セッション毎に処理
    for s in range(param.SESSION_NUM):
        # 使用する特徴量の選択
        if param.TARGET_SIGNAL == "HBX":
            # 左右のtotalHBのデータのみを抽出 （元データに干渉しないよう、arrayを新しく生成）
            signal1 = np.array(signal_data_per_session[s][:,1].T) # 1次元array型
            signal2 = np.array(signal_data_per_session[s][:,4].T) # 1次元array型
        if param.TARGET_SIGNAL == "HBX1":
            # 左右のtotalHB(深さ1cm)のデータのみを抽出 （元データに干渉しないよう、arrayを新しく生成）
            signal1 = np.array(signal_data_per_session[s][:,2].T) # 1次元array型
            signal2 = np.array(signal_data_per_session[s][:,5].T) # 1次元array型
        if param.TARGET_SIGNAL == "HBX3":
            # 左右のtotalHB(深さ3cm)のデータのみを抽出 （元データに干渉しないよう、arrayを新しく生成）
            signal1 = np.array(signal_data_per_session[s][:,3].T) # 1次元array型
            signal2 = np.array(signal_data_per_session[s][:,6].T) # 1次元array型
       
        # 信号列の可視化
        if False:  
            visualize_signal(subject_id, s, signal1, signal2, param, _range="free")

        # 各データの前処理
        preprocessed_signal_array1 = preprocessing(signal1, param)
        preprocessed_signal_array2 = preprocessing(signal2, param)
        
        # ノイズ判定
        noisy_signal_array1 = detect_noise(preprocessed_signal_array1, param)
        noisy_signal_array2 = detect_noise(preprocessed_signal_array2, param)
        # 合算
        noisy_signal_array  = np.asarray(np.logical_or(noisy_signal_array1, noisy_signal_array2), dtype=int)
#        print noisy_signal_array
        
        # 信号列の可視化
        if False:
            visualize_signal(subject_id, s, preprocessed_signal_array1, preprocessed_signal_array2, 
                             param, _range="free",
                             noise = noisy_signal_array)
            
        # データから特徴量の抽出・ノイズ部位の除去
        feature_array1, used_flag_list = extract_features(preprocessed_signal_array1, log_per_session[s], param, noisy_signal = noisy_signal_array)
        feature_array2, used_flag_list = extract_features(preprocessed_signal_array2, log_per_session[s], param, noisy_signal = noisy_signal_array)
        #print used_flag_list
        # 左右データの結合 （列方向での結合）
        feature_array = np.c_[feature_array1, feature_array2]

        # ノイズ判定に基づくラベルの削減
        used_label_list = [label_list_per_session[s][i] for i in range(len(label_list_per_session[s])) if used_flag_list[i]]
        
        # ラベルの使用に合わせたデータ削減
        new_label_list_per_session = []
        new_feature_array_list = []
        for i in range(len(used_label_list)):
            if used_label_list[i] == None:
                pass
            else:
                new_label_list_per_session.append(used_label_list[i])
                new_feature_array_list.append(feature_array[i])
            
        # 全セッションデータの融合
        all_label_list.extend(new_label_list_per_session)
        all_feature_list.append(np.vstack(new_feature_array_list))

        # セッション毎の特徴量の可視化
        if False: 
            visualize_feature(subject_id, s, np.vstack(new_feature_array_list), 
                              new_label_list_per_session, param)
            visualize_feature(subject_id, s, np.vstack(new_feature_array_list), 
                              new_label_list_per_session, param, print_all = True)
        
    # 行方向にsingalのarrayを連結
    all_feature_array = np.vstack(all_feature_list)
    # 特徴量の次元数を定数化
    feature_dimension_size = len(all_feature_array[0])

    # 全セッションの特徴量の可視化
    if False:
        visualize_feature(subject_id, "all", all_feature_array, all_label_list, param)
        visualize_feature(subject_id, "all", all_feature_array, 
                          all_label_list, param, print_all = True)

    if False:
        # 使用する特徴量の一次出力
        cc = csv.writer(open("tmp.csv", 'wb'))
        session_sample_id_list = []
        for s in range(param.SESSION_NUM):
            session_sample_id_list.extend([s+1 for _ in range(len(all_feature_list[s]))])
        ses_array = np.reshape(np.array(session_sample_id_list), (len(session_sample_id_list),1))
        lab_array = np.reshape(np.array(all_label_list), (len(all_label_list),1))
        ret_array = np.c_[ses_array, lab_array, all_feature_array]
        ret_list = ret_array.tolist()
        cc.writerows(ret_list)

    # 画像のみ描画したい場合はここでエラーを発生させると良い。
    #raise MyError("Next subject!")
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
        test_feature_array = np.array(dataset_feature_array[k], dtype=np.float32)

        # トレーニングデータ
        training_label_list = []
        training_feature_list = []
        for j in range(param.N_FOLD):
            if k != j:
                training_label_list.extend(dataset_label_list[j])
                training_feature_list.append(dataset_feature_array[j])
        #print training_feature_list
        training_feature_array = np.vstack(training_feature_list)
            
        # リサンプリング処理
        print "resampling..."
        resampled_training_label_list, resampled_training_feature_array = recon.resampling(training_label_list, 
                                                                                     training_feature_array, param)
        
        # トレーニングデータの並び替え
        mixed = np.array(resampled_training_label_list).reshape((len(resampled_training_label_list),1))
        mixed = np.c_[mixed, resampled_training_feature_array]            
        np.random.shuffle(mixed)
        resampled_training_label_array = mixed[:,0].flatten()
        resampled_training_feature_array = mixed[:,1:]

        # array型への変換
        # 特徴量
        resampled_training_feature_array = np.array(resampled_training_feature_array, dtype=np.float32)
        test_feature_array = np.array(test_feature_array, dtype=np.float32)
        
        # ラベル
        if param.LEARNING_TYPE == "class":  
            resampled_training_label_array = np.array(resampled_training_label_array, dtype=np.int32)
            test_label_array = np.array(test_label_list, dtype=np.int32)
        elif param.LEARNING_TYPE == "regression":
            resampled_training_label_array = np.array(resampled_training_label_array, dtype=np.float32)
            test_label_array = np.array(test_label_list, dtype=np.float32)

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
        
        # LDA
        if param.LEARNING_METHOD == "LDA":
            print "LDA"
            # クラス分類型
            if param.LEARNING_TYPE == "class":
                print "class"
                
                # トレーニング & 識別
                print "training & test..."
                precision, result_label_list, result_probability_list, lda_obj = mclda.multiclass_lda(resampled_training_feature_array, resampled_training_label_array, test_feature_array, test_label_array)
                
                # 識別結果の格納
                default_accuracy += precision
                all_test_label_list.extend(test_label_list)
                all_result_label_list.extend(result_label_list)
                all_result_loss_list.extend(result_probability_list)        
        # LDA
        if param.LEARNING_METHOD == "SVM":
            print "SVM"
            # クラス分類型
            if param.LEARNING_TYPE == "class":
                print "class"
                
                # トレーニング & 識別
                print "training & test..."
                precision, result_label_list, result_probability_list, svm_obj = mcsvm.multiclass_svm(resampled_training_feature_array, resampled_training_label_array, test_feature_array, test_label_array)
                
                # 識別結果の格納
                default_accuracy += precision
                all_test_label_list.extend(test_label_list)
                all_result_label_list.extend(result_label_list)
                all_result_loss_list.extend(result_probability_list)        
        # N-FOLD終わり
        
    # 全体識別率
    default_accuracy /= param.N_FOLD
    default_loss /= param.N_FOLD
    
    if PRINT_FLAG:
        print 
        if param.LEARNING_TYPE == "class":
            if param.LEARNING_METHOD == "NN":
                print "acc:", default_accuracy, ", loss:", default_loss
            elif param.LEARNING_METHOD == "LDA":
                print "acc:", default_accuracy
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
        if param.LEARNING_METHOD == "NN":
            logging_result_class(subject_id, result_filename, default_accuracy, default_loss, precision, recall, f_measure, dist, corr, p_value, result_mat, all_test_label_list, all_result_label_list, all_result_loss_list)
        elif param.LEARNING_METHOD == "LDA":
            # default_lossがないので0で代用
            # all_result_loss_listがprobabilityのリストであることに注意
            logging_result_class(subject_id, result_filename, default_accuracy, 0,  precision, recall, f_measure, dist, corr, p_value, result_mat, all_test_label_list, all_result_label_list, all_result_loss_list)
        elif param.LEARNING_METHOD == "SVM":
            # default_lossがないので0で代用
            # all_result_loss_listがprobabilityのリストであることに注意
            logging_result_class(subject_id, result_filename, default_accuracy, 0,  precision, recall, f_measure, dist, corr, p_value, result_mat, all_test_label_list, all_result_label_list, all_result_loss_list)
    
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

    

def detect_noise(signal, param):
    """
    ノイズの判定
    @param signal: fNIRS信号 1次元array型
    @param param: パラメータオブジェクト
    
    @return: noisy_signal ノイズ領域を1、非ノイズ領域を0とした1次元array  
    """
    # ノイズ判定範囲の設定
    width = param.NOISY_T_RANGE/2
    
    # ノイズ領域を格納するarray
    noisy_signal = np.array([0 for i in range(len(signal))])
    # 最初と最後はノイズ領域にしておく
    noisy_signal[:width] = 1
    noisy_signal[len(signal)-width:] = 1
    
    for i in range(width, len(signal)-width):
        if max(signal[i-width:i+width])-min(signal[i-width:i+width]) > param.NOISY_THRESHOLD:
            noisy_signal[i:i+param.AFTER_NOISE] = 1
    
    return noisy_signal

def preprocessing(signal, param):
    """
    各シグナルの前処理（確認済）
    @param singal: fNIRS信号 1次元array型
    @param param: パラメータモジュール  
    
    @return: 前処理終了後のシグナル値 1次元array型（特徴次元数）
    """            
    ###### データの前処理
    # バンドパスフィルタ
    if param.FILTER_TYPE == "butter":
        preprocessed_signal = prba.butter_bandpass_filter(signal, param.BPF_BAND[0], param.BPF_BAND[1], 
                                                          param.FS, order = param.BUTTER_WORTH_ORDER)
    else: # フィルタ処理無し
        preprocessed_signal = signal 
    
    # 平滑化
    if param.SMOOTHING_TYPE:
        preprocessed_signal = prsm.smoothing(preprocessed_signal, param.SMOOTHING_LENGTH)
    else: # 平滑化処理無し
        pass
    
    # 1次微分
    if param.DIFF >= 1:
        preprocessed_signal_n = np.roll(preprocessed_signal, -1)
        preprocessed_signal = preprocessed_signal_n - preprocessed_signal
    # 2次微分
    elif param.DIFF == 2:
        preprocessed_signal_n = np.roll(preprocessed_signal, -1)
        preprocessed_signal_p = np.roll(preprocessed_signal, 1)
        preprocessed_signal = (preprocessed_signal_n - preprocessed_signal) - (preprocessed_signal - preprocessed_signal_p)
    
    return preprocessed_signal

def extract_features(preprocessed_signal, task_start_index, param, noisy_signal = []):
    """
    前処理後のシグナルからの特徴量抽出
    @param preprocessed_signal: 前処理後の特徴量
    @param task_start_index: タスク開始のインデックスデータ 
    @param param: パラメータモジュール
    @keyword noisy_signal: ノイズ領域array 
    @return: 特徴量 2次元array型（サンプル数×特徴次元数）, user_flag_listその特徴量が使用されたか否か  
    """
    # タスク期間中のデータの切り出し
    signal_array = extract_task_signal.extract_task_signal(preprocessed_signal, task_start_index, param.TASK_DURATION)
    # ノイズ領域に重複している場合はそのデータを除去
    used_flag_list = [] # 使用されたか否か
    if noisy_signal != []:
        noisy_array = extract_task_signal.extract_task_signal(noisy_signal, task_start_index, param.TASK_DURATION)
        noise_reduced_signal_list = []
        for i in range(len(signal_array)):
            if sum(noisy_array[i]) > 0:
                used_flag_list.append(False)
            else:
                used_flag_list.append(True)
                noise_reduced_signal_list.append(signal_array[i])
        signal_array = np.vstack(noise_reduced_signal_list)
    
    ###### 特徴量抽出
    # 1秒おきの抽出
    if param.FEATURE_TYPE == "SLICE":
        sliced_singal_list = []
        for i in range(len(signal_array)):
            new_sig = slice_signal(signal_array[i], param.SLICE_START_LAG, param.SLICE_RATE)
            # 0点補正
            new_sig = normalize_0point(new_sig)
            sliced_singal_list.append(new_sig)
        feature_array = np.asarray(sliced_singal_list, dtype=np.float32)
    
    # SSとSMの計算
    elif param.FEATURE_TYPE == "SSSM":
        new_feature_list = []
        for i in range(len(signal_array)):
            ss, sm = get_SS_SM(signal_array[i,], param.SSSM_START, param.SSSM_WINDOW)
            new_feature_list.append([ss, sm])
        feature_array = np.asarray(new_feature_list, dtype=np.float32)

    # MeanとSkewnessの計算
    elif param.FEATURE_TYPE == "SMSK":
        new_feature_list = []
        for i in range(len(signal_array)):
            sm, sk = get_SM_SK(signal_array[i,])
            new_feature_list.append([sm, sk])
        feature_array = np.asarray(new_feature_list, dtype=np.float32)
    
    else:
        raise NameError("Parameter miss: FEATURE TYPE not exist: " + str(param.FEATURE_TYPE))
    
    return feature_array, used_flag_list

def relabeling(label_list_per_session, param):
    """
    リラベル
    @param label_list_per_session: ラベルリスト（セッション×各ラベル）
    @param param: パラメータモジュール 
    
    @return: new_label_list_per_session 付け直されたラベルのリスト
    """
    # 新しいラベル
    new_label_list_per_session = []

    # Regression用のラベル
    if param.RELABEL_METHOD == "scaling":
        print "Realabel isn't performed...*0.1"
        for s in range(param.SESSION_NUM):
            new_label_list_per_session.append([l*0.1-0.5 for l in label_list_per_session[s]])
    
    # マッピングに従う固定リラベル
    if param.RELABEL_METHOD == "mapping":
        print "Classess are groupd by a mapping list."
        for s in range(param.SESSION_NUM):
            new_label_list_per_session.append(rela.relabel_basedon_mapping(label_list_per_session[s], param.RELABEL_MAPPING))

    # 分散最小化
    elif param.RELABEL_METHOD == "var":
        print "Classess are groupd by varinace values."
        # 全セッションデータの結合
        reduced_label_list_per_session = reduce(lambda x,y: x+y, label_list_per_session)
        new_label_list = rela.relabel_minimize_variance(reduced_label_list_per_session, param.CLASS_NUM)
        counter = 0
        for i in range(len(label_list_per_session)):
            new_label_list_per_session.append([])
            for _ in range(len(label_list_per_session[i])):
                new_label_list_per_session[i].append(new_label_list[counter])
                counter += 1
#    print label_list_per_session
#    print new_label_list_per_session
#    quit()
    
    return new_label_list_per_session
        
        
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
    
def visualize_signal(subject_id, session_num, signal1, signal2, param, 
                     _range="static", noise = None):
    """
    あるセッションに置ける前処理後の信号列の可視化（左右双方）
    @param subect_id: 被験者ID 
    @param session_num: セッション番号
    @param signal1, signal2: 処理された信号 
    @param param: パラメータモジュール  
    @keyword _range: staticだと-0.25〜0.25、freeだと自動決定 
    @keyword noise: ノイズ情報のarray
    """
    # 可視化
#    import matplotlib.mpl.rcParams as rcp
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(12,5))
    plt.clf()
    plt.rcParams['font.size'] = 10
    t = np.linspace(0, param.TOTAL_SAMPLE_NUM/param.FS, param.TOTAL_SAMPLE_NUM, endpoint=False)
    plt.plot(t, signal1[:param.TOTAL_SAMPLE_NUM], label='signal_left_%s'%(param.TARGET_SIGNAL))
    plt.plot(t, signal2[:param.TOTAL_SAMPLE_NUM], label='signal_right_%s'%(param.TARGET_SIGNAL))
    for i in range(25):
        plt.axvspan(15+i*25, 25+25*i, color='red', alpha=0.5)
    # ノイズ部分を編みかけ
    if noise != None: 
        st = -1
        for i in range(param.TOTAL_SAMPLE_NUM):
            if st == -1 and noise[i] == 1:
                st = i
            elif st != -1 and noise[i] == 0:
                plt.axvspan(st/param.FS, i/param.FS, color='grey', alpha=0.5)
                st = -1
        if st != -1:
            plt.axvspan(st/param.FS, param.TOTAL_SAMPLE_NUM/param.FS, color='grey', alpha=0.5)
            st = -1
        
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    if _range == "static":
        plt.ylim(-0.25, 0.25)
    plt.savefig("analysis/visualization1/"+subject_id+"_"+param.TARGET_SIGNAL+"_s"+str(session_num)+".png")

def visualize_feature(subject_id, session_num, signal_array, label_list, param, print_all=False):
    """
    あるセッションに置ける前処理後の信号列の可視化（左右双方）
    @param subect_id: 被験者ID 
    @param session_num: セッション番号
    @param signal_array: 処理された信号 2次元array型
    @param label_list: セッション毎のラベルリスト 
    @param param: パラメータモジュール 
    @keyword print_all: 平均だけでなく全データを描画する  
    """
    import matplotlib.pyplot as plt

    if session_num == "all":
        # 結合
        label_array = np.array(label_list).flatten()
        new_array = np.hstack((label_array.reshape(len(label_array),1), signal_array))
    else:
        new_array = np.hstack((np.array(label_list).reshape(len(label_list),1), signal_array))
        
    # 分化
    new_label_array = [[] for _ in range(len(param.CLASS_LIST))]
    for na in new_array:
        new_label_array[int(na[0])].append(na)

    # 可視化
    plt.figure(1, figsize=(12,5))
    plt.clf()
    dim_size = len(new_array[0])

    color_list = ['blue', 'cyan', 'green', '#FFA500', 'red']

    # 統計処理および可視化
    if param.FEATURE_TYPE == "SLICE":
        for c in range(len(param.CLASS_LIST)):
            if new_label_array[c] != []:
                if print_all:
                    new_label_array_c = np.asarray(new_label_array[c])
                    t = np.asarray(range(dim_size-1))
                    # 個々のデータの描画
                    for d in range(len(new_label_array_c)):
                        plt.plot(t, new_label_array_c[d, 1:dim_size], 
                                 color = color_list[c], linewidth=0.5)
                    # 平均・標準偏差の描画
                    neo_averaged = np.mean(np.asarray(new_label_array[c]), axis=0)
                    neo_variance = np.std(np.asarray(new_label_array[c]), axis=0)
                    plt.errorbar(t, neo_averaged[1:dim_size], 
                                 yerr=neo_variance[1:dim_size], 
                                 color = color_list[c], linewidth=2,
                                 label='signal_class%s'%(str(param.CLASS_LIST[c])))
                else:
                    t = np.asarray(range(dim_size-1))
                    # 平均・標準偏差の描画
                    neo_averaged = np.mean(np.asarray(new_label_array[c]), axis=0)
                    neo_variance = np.std(np.asarray(new_label_array[c]), axis=0)
                    plt.errorbar(t, neo_averaged[1:dim_size], 
                                 yerr=neo_variance[1:dim_size], 
                                 color = color_list[c], linewidth=2,
                                 label='signal_class%s'%(str(param.CLASS_LIST[c])))

    elif param.FEATURE_TYPE in ["SSSM", "SMSK"]:
        # 各次元のスケーリング用の絶対値最大値のarray
        new_array_max = np.max(abs(new_array), axis=0)
        new_array_max[new_array_max==0] = 1 # 0で割らないように工夫
        
        for c in range(len(param.CLASS_LIST)):
            if new_label_array[c] != []:
                if print_all:
                    new_label_array_c = np.asarray(new_label_array[c])
                    new_label_array_c /= new_array_max
                    t = np.asarray(range(1,dim_size))#np.linspace(1, dim_size-1, dim_size, endpoint=False)
                    # 個々のデータの描画
                    for d in range(len(new_label_array_c)):
                        plt.plot(t, new_label_array_c[d][1:dim_size+1], 
                                 color = color_list[c])
                    # 平均・標準偏差の描画
                    neo_averaged = np.mean(new_label_array_c, axis=0)
                    neo_variance = np.std(new_label_array_c, axis=0)
                    t = np.asarray(range(1,dim_size))#np.linspace(1, dim_size-1, dim_size, endpoint=False)
                    plt.errorbar(t, neo_averaged[1:dim_size], 
                                 yerr=neo_variance[1:dim_size], 
                                 color = color_list[c],
                                 label='signal_class%s'%(str(param.CLASS_LIST[c])))
                else:
                    new_label_array_c = np.asarray(new_label_array[c])
                    new_label_array_c /= new_array_max
                    # 平均・標準偏差の描画
                    neo_averaged = np.mean(new_label_array_c, axis=0)
                    neo_variance = np.std(new_label_array_c, axis=0)
                    t = np.asarray(range(1,dim_size))#np.linspace(1, dim_size-1, dim_size, endpoint=False)
                    plt.errorbar(t, neo_averaged[1:dim_size], 
                                 yerr=neo_variance[1:dim_size], 
                                 color = color_list[c],
                                 label='signal_class%s'%(str(param.CLASS_LIST[c])))
        plt.xlim(0.0,dim_size+1)
        plt.ylim(-1.3, 1.3)

    # 可視化
    plt.xlabel('time (seconds)')
    plt.grid(True)
#    plt.axis('tight')
    plt.legend(loc='upper left')
    if print_all:
        plt.savefig("analysis/visualization3/"+subject_id+"_"+param.TARGET_SIGNAL+"_"+str(session_num)+".png")
    else:
        plt.savefig("analysis/visualization2/"+subject_id+"_"+param.TARGET_SIGNAL+"_"+str(session_num)+".png")
    
if __name__=="__main__":
    repeat_main()