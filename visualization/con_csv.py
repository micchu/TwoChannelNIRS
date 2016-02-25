#coding:utf-8

'''
Created on 2016/02/25

@author: misato
'''
import codecs
import csv

def read_table(target_filename):
    """
    比較するデータの読み出し
    @param target_filename: データ読み出し元のファイル名  
    """
    # 読み込み
    fp = open(target_filename, 'Ur')
    cr = csv.reader(fp, dialect = "excel", delimiter = ",")
    # ヘッダ
    cr.next()
    # 読み込み
    parameter_list = []
    filename_list = []
    for c in cr:
        parameter_list.append([float(a_c) if a_c.replace(".","").isdigit() else a_c for a_c in c[0:5]])
        filename_list.append(c[5])
#        print [float(a_c) if a_c.replace(".","").isdigit() else a_c for a_c in c[0:5]]
#        quit()
        
    return parameter_list, filename_list

def read_data(target_filename, target_data):
    """
    @param target_filename: データファイル 
    @param target_data: 比較対象のデータ ("all"の場合は主要な結果9個を返す)
    @return; IDリスト，ターゲットのデータリスト 
    """
    # 読み込み
    fp = open(target_filename, 'Ur')
    cr = csv.reader(fp, dialect = "excel", delimiter = ",")
    cr.next()
    # ヘッダ
    header_list = ["subject_id","accuracy","loss(LDAなどではなし)","average_precition",
                   "average_recall","average_Fmeasure","average_distance","corr","p_value"]
    if target_data != "all":
        index = header_list.index(target_data)
    # 読み込み
    subject_id_list = []
    data_list = []
    for c in cr:
        subject_id_list.append(c[0])
        if target_data != "all":
            data_list.append(float(c[index]))
        else:
            data_list.append([float(a_c) if a_c.replace(".","").isdigit() else a_c for a_c in c[:9]])

    return subject_id_list, data_list

def generate_pivot(header_list, parameter_list, data_list, output_filename):
    """
    統一ピボット形式への変換
    
    @param header_list: ヘッダのリスト <1次元リスト m parameters + j result_values>
    @param parameter_list: パラメータのリスト <2次元リスト n patterns * m parameters>
    @param data_list: データリスト <3次元リスト n patterns * i subjects * j result_values>
    @param output_filename: 出力先ファイル
     
    @return: ピボット形式のデータ <2次元リスト (m parameters + j result_values) * ?>
    """
    # 格納先
    merge_list = [header_list]
    
    # パラメータ毎にデータの読み出し
    for n in range(len(parameter_list)):
        # パラメータをコピー
        for i in range(len(data_list[n])):
            # 被験者毎のデータを追記
            a_container = parameter_list[n][:]
            a_container.extend(data_list[n][i])
            merge_list.append(a_container)
#        print merge_list
#        quit(9)
            
    cw = csv.writer(open(output_filename, 'wb'), quoting = csv.QUOTE_NONE)
    cw.writerows(merge_list)
    