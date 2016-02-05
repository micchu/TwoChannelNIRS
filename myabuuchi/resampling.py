# coding: utf-8
'''
Created on 2015/10/16
@author: myabuuchi
'''


# In[4]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys
import pandas as pd
#from svm import * 
#import tools.subset as sub
from collections import defaultdict
import random
import numpy.random as nr
import Chainer_MLP as MLP



#アンケート結果の読み込み
def read_data_anq(fname):
    data_anq=[]
    
    #5class分類
    data=pd.read_csv("/Users/myabuuchi/workspace/task_2015/anquate/"+fname+"_anq_new.csv")
    '''
    #3class分類
    data=pd.read_csv("/Users/myabuuchi/workspace/task_2015/anquate/"+fname+"_anq_3class.csv")
    '''
    for i in range(4):
        for k in range(i*26,i*26+25):
            select_data=data.ix[k,3]
            #print select_data
            data_anq.append(select_data)
    return data_anq

#計測データの読み込み
def read_data_preprocessed(fname):
    data_preprocessed=np.zeros((100,20))
    count=1
    for count in range(1,5):
        
        #ローデータの読み込み
        data_pre=pd.read_csv("/Users/myabuuchi/workspace/task_2015/preprocessed/"+fname+str(count)+"_preprocessed.csv",header=None)
        data_pre.values
        data_preprocessed[(count-1)*25:(count-1)*25+25,:]=np.array(data_pre)
        '''
        #1000倍したデータの読み込み
        data_pre=np.genfromtxt("/Users/myabuuchi/workspace/task_2015/preprocessed/"+fname+str(count)+"_preprocessed_change.csv",delimiter=",",usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19),skiprows=30)
        data_preprocessed[(count-1)*25:(count-1)*25+25,:]=np.array(data_pre)
        '''
        #print data_preprocessed
        #print data_preprocessed.shape
        #quit()
    return data_preprocessed

#計測データとアンケート結果の結合
def conect_blood_anq(data_anq,data_preprocessed):
    data_connect=np.zeros((100,21))
    data_connect[:,0]=data_anq
    data_connect[:,1:21]=data_preprocessed
    return data_connect

#データセットを作成する関数
#5class分類
def make_dataset(data_connect):
    print "data_connect"
    print data_connect
    dataset=data_connect
    count_value1=0
    count_value2=0
    count_value3=0
    count_value4=0
    count_value5=0
    #各評価毎にデータを分ける
    for i in range(100):
        if dataset[i,0]==1:
            if count_value1==0:
                value1=np.array([dataset[i]])
                count_value1+=1
            else:
                value1=np.r_[value1,[dataset[i]]]
        elif dataset[i,0]==2:
            if count_value2==0:
                value2=np.array([dataset[i]])
                count_value2+=1
            else:
                value2=np.r_[value2,[dataset[i]]]
        elif dataset[i,0]==3:
            if count_value3==0:
                value3=np.array([dataset[i]])
                count_value3+=1
            else:
                value3=np.r_[value3,[dataset[i]]]
        elif dataset[i,0]==4:
            if count_value4==0:
                value4=np.array([dataset[i]])
                count_value4+=1
            else:
                value4=np.r_[value4,[dataset[i]]]
        else:
            if count_value5==0:
                value5=np.array([dataset[i]])
                count_value5+=1
            else:
                value5=np.r_[value5,[dataset[i]]]
    #データセットをランダムにシャッフルする
#    print value1.shape
#    print value2.shape
#    print value3.shape
#    print value4.shape
#    print value5.shape
    #quit()
    nr.shuffle(value1)
    nr.shuffle(value2)
    nr.shuffle(value3)
    nr.shuffle(value4)
    nr.shuffle(value5)
    #各評価を１つにまとめる
    value=np.array(value1)
    value=np.r_[value,value2]
    value=np.r_[value,value3]
    value=np.r_[value,value4]
    value=np.r_[value,value5]
#    print value.shape
    #各評価のデータ群を10分割するために１つずつ順番に10コの箱に入れていく
    divide_box=[[]for i in range(10)]
    for i in range(10):
        count=0
        for j in range(10):
            if count==10:
                count=0
            divide_box[count].append(value[i*10+j])
            count+=1
#    print "len"
#    print len(divide_box)
#    print len(divide_box[0])
#    print divide_box[0]
#    quit()

    return divide_box

#3class分類
def make_dataset_3class(data_connect):
    dataset=data_connect
    count_value1=0
    count_value2=0
    count_value3=0
    #各評価毎にデータを分ける
    for i in range(100):
        if dataset[i,0]==1:
            if count_value1==0:
                value1=np.array([dataset[i]])
                count_value1+=1
            else:
                value1=np.r_[value1,[dataset[i]]]
        elif dataset[i,0]==2:
            if count_value2==0:
                value2=np.array([dataset[i]])
                count_value2+=1
            else:
                value2=np.r_[value2,[dataset[i]]]
        else:
            if count_value3==0:
                value3=np.array([dataset[i]])
                count_value3+=1
            else:
                value3=np.r_[value3,[dataset[i]]]  
    #データセットをランダムにシャッフルする
    nr.shuffle(value1)
    nr.shuffle(value2)
    nr.shuffle(value3)
    #各評価を１つにまとめる
    value=np.array(value1)
    value=np.r_[value,value2]
    value=np.r_[value,value3]
    print value.shape
    #各評価のデータ群を10分割するために１つずつ順番に10コの箱に入れていく
    divide_box=[[]for i in range(10)]
    for i in range(10):
        count=0
        for j in range(10):
            if count==10:
                count=0
            divide_box[count].append(value[i*10+j])
            count+=1
    return divide_box


#リサンプリング処理
#5class分類
def resampling_data(divide_box,fname):
    """
    @param divide_box: 3次元、N-foldに分割された後のデータ (fold数 × サンプル × (特徴量数+ラベル))
        例：10 fold, 10 sample, 20 feature + 1 label
    @param fname: 被験者名 
    """
    list_acc_loss=[]
    #リサンプリングされたデータを入れる箱を作成する
    data_resampling=[]
    divide_data=np.array(divide_box)
    nr.shuffle(divide_data)
    #print divide_data
    #print divide_data.shape
    #10foldするために各データ群を1回ずつテストデータに入力し残りをリサンプリングしてトレーニングデータに入力する
    for i in range(10):
        #テストデータを入れる箱を作成し，テストデータに各データ群を入れていく
        _test=divide_data[i]
        #print _test
        #print _test.shape
        x_test=_test[:,1:21].astype(np.float32)
        y_test=_test[:,0].astype(np.int32)-1
        #y_test=_test[:,0].astype(np.int32)
        #print x_test
        #各評価毎に再分割するために各評価の値を見やすいように配列化する
        _remaining_data=np.delete(divide_data,i,0)
        #print remaining_data.shape
        remaining_data=np.array(_remaining_data)
#        print remaining_data
        #print remaining_data
        #print remaining_data.shape
        #quit()
        remain_data=[]
        #データを配列化する
        for k in remaining_data:
            for n in k:
                remain_data.append(n.tolist())
        remained_data=np.array(remain_data)
        #print remained_data
        #print remained_data.shape
        #quit()
        value1_count=0
        value2_count=0
        value3_count=0
        value4_count=0
        value5_count=0
        #各評価毎に分割する
        for m in range(remained_data.shape[0]):
            #print remained_data.shape
            if remained_data[m,0]==1:
                if value1_count==0:
                    data_value1=np.array([remained_data[m]])
                    value1_count+=1
                else:
                    data_value1=np.r_[data_value1,[remained_data[m]]]
            elif remained_data[m,0]==2:
                if value2_count==0:
                    data_value2=np.array([remained_data[m]])
                    value2_count+=1
                else:
                    data_value2=np.r_[data_value2,[remained_data[m]]]
            elif remained_data[m,0]==3:
                if value3_count==0:
                    data_value3=np.array([remained_data[m]])
                    value3_count+=1
                else:
                    data_value3=np.r_[data_value3,[remained_data[m]]]
            elif remained_data[m,0]==4:
                if value4_count==0:
                    data_value4=np.array([remained_data[m]])
                    value4_count+=1
                else:
                    data_value4=np.r_[data_value4,[remained_data[m]]]
            else:
                if value5_count==0:
                    data_value5=np.array([remained_data[m]])
                    value5_count+=1
                else:
                    data_value5=np.r_[data_value5,[remained_data[m]]]
        #データに
        data_value=[]
        data_value.append(data_value1)
        data_value.append(data_value2)
        data_value.append(data_value3)
        data_value.append(data_value4)
        data_value.append(data_value5)
        #print len(data_value)
        #quit()
        data_resampling=[]
        for j in range(5):
            selected_ave=[]
            num_sum=[]
            data_value=np.array(data_value)
            #print data_value.shape
            data_value.shape
            
            for ix in xrange(data_value.shape[0]):
                data_value[ix]=np.array(data_value[ix])
            for ix in xrange(data_value.shape[0]):
                #print data_value[ix].shape[0]
                data_value[ix].shape[0]
            #print data_value
            data_value=np.array(data_value)
            _data_value=data_value[j]
            length = _data_value.shape[0]
            #quit()
            
            for p in range(length):
                num=[]
                _data_stock=[]
                for l in range(5):
                    rand = random.randint(0,length-1)
                    while(rand in num):
                        rand = random.randint(0,length-1)
                    _data_stock.append(_data_value[rand])
                    #print _data_stock
                    #print len(_data_stock)
                    num.append(rand)
                    num_sum.append(num)
                    l+=1
                data_stock=np.array(_data_stock)
                #print data_stock
                selected_ave = sum(data_stock) / 5
                #print selected_ave
                #quit()
                data_resampling.append(selected_ave)
#                n+=1
#                p+=1
        data_resampled=np.array(data_resampling)
        #print data_resampled
        #print data_resampled.shape
        #quit()
        '''
        #トレーニングデータ180コ
        data_increased=data_resampled
        data_increased=np.append(data_increased,remained_data)
        data_increased=data_increased.reshape(180,21)
        #print data_increased.shape
        #print data_increased
        #quit()
        #トレーニングデータに入れる
        x_train = np.array(data_increased[:, 1:21]).astype(np.float32)
        y_train = np.array(data_increased[:, 0]).astype(np.int32)-1
        '''
        #トレーニングデータに入れる
        x_train = np.array(data_resampled[:, 1:21]).astype(np.float32)
        y_train = np.array(data_resampled[:, 0]).astype(np.int32)-1
        
        #y_train = np.array(data_resampled[:, 0]).astype(np.int32)
        #print x_trainacc_loss=MLP.learning(x_train,y_train,x_test,y_test,n_fold=i,fname=fname)
        acc_loss=MLP.learning(x_train,y_train,x_test,y_test,n_fold=i,fname=fname)
        list_acc_loss.append(acc_loss)
    list_acc_loss=np.array(list_acc_loss)
    print list_acc_loss.shape
    list_acc_loss=list_acc_loss.reshape(100,5)
    #quit()
    
    #3class分類
def resampling_data_3class(divide_box,fname):
    list_acc_loss=[]
    #リサンプリングされたデータを入れる箱を作成する
    data_resampling=[]
    divide_data=np.array(divide_box)
    nr.shuffle(divide_data)
    #print divide_data
    #print divide_data.shape
    #10foldするために各データ群を1回ずつテストデータに入力し残りをリサンプリングしてトレーニングデータに入力する
    for i in range(10):
        #テストデータを入れる箱を作成し，テストデータに各データ群を入れていく
        _test=divide_data[i]
        #print _test
        #print _test.shape
        x_test=_test[:,1:21].astype(np.float32)
        y_test=_test[:,0].astype(np.int32)-1
        #y_test=_test[:,0].astype(np.int32)
        #print x_test
        #各評価毎に再分割するために各評価の値を見やすいように配列化する
        remaining_data=np.delete(divide_data,i,0)
        #print remaining_data.shape
        remaining_data=np.array(remaining_data)
        #print remaining_data
        #print remaining_data.shape
        remain_data=[]
        #データを配列化する
        for k in remaining_data:
            for n in k:
                remain_data.append(n.tolist())
        remained_data=np.array(remain_data)
        #print remained_data
        #print remained_data.shape
        value1_count=0
        value2_count=0
        value3_count=0
        #各評価毎に分割する
        for m in range(remained_data.shape[0]):
            #print remained_data.shape
            if remained_data[m,0]==1:
                if value1_count==0:
                    data_value1=np.array([remained_data[m]])
                    value1_count+=1
                else:
                    data_value1=np.r_[data_value1,[remained_data[m]]]
            elif remained_data[m,0]==2:
                if value2_count==0:
                    data_value2=np.array([remained_data[m]])
                    value2_count+=1
                else:
                    data_value2=np.r_[data_value2,[remained_data[m]]]
            else:
                if value3_count==0:
                    data_value3=np.array([remained_data[m]])
                    value3_count+=1
                else:
                    data_value3=np.r_[data_value3,[remained_data[m]]]
        #データに
        data_value=[]
        data_value.append(data_value1)
        data_value.append(data_value2)
        data_value.append(data_value3)
        #print len(data_value)
        #quit()
        data_resampling=[]
        for j in range(3):
            selected_ave=[]
            num_sum=[]
            data_value=np.array(data_value)
            #print data_value.shape
            data_value.shape
            
            for ix in xrange(data_value.shape[0]):
                data_value[ix]=np.array(data_value[ix])
            for ix in xrange(data_value.shape[0]):
                #print data_value[ix].shape[0]
                data_value[ix].shape[0]
            #print data_value
            data_value=np.array(data_value)
            _data_value=data_value[j]
            length = _data_value.shape[0]
            #quit()
            
            for p in range(length):
                num=[]
                _data_stock=[]
                for l in range(3):
                    rand = random.randint(0,length-1)
                    while(rand in num):
                        rand = random.randint(0,length-1)
                    _data_stock.append(_data_value[rand])
                    #print _data_stock
                    #print len(_data_stock)
                    num.append(rand)
                    num_sum.append(num)
                    l+=1
                data_stock=np.array(_data_stock)
                #print data_stock
                selected_ave = sum(data_stock) / 5
                #print selected_ave
                #quit()
                data_resampling.append(selected_ave)
                n+=1
                p+=1
        data_resampled=np.array(data_resampling)
        #print data_resampled
        print data_resampled.shape
        #トレーニングデータに入れる
        x_train = np.array(data_resampled[:, 1:21]).astype(np.float32)
        y_train = np.array(data_resampled[:, 0]).astype(np.int32)-1
        #y_train = np.array(data_resampled[:, 0]).astype(np.int32)
        #print x_trainacc_loss=MLP.learning(x_train,y_train,x_test,y_test,n_fold=i,fname=fname)
        acc_loss=MLP.learning(x_train,y_train,x_test,y_test,n_fold=i,fname=fname)
        list_acc_loss.append(acc_loss)
    list_acc_loss=np.array(list_acc_loss)
    print list_acc_loss.shape
    list_acc_loss=list_acc_loss.reshape(100,5)
    
#リサンプリングなし
def no_resampling_data(divide_box,fname):
    list_acc_loss=[]
    divide_data=np.array(divide_box)
    nr.shuffle(divide_data)
    #print divide_data
    #print divide_data.shape
    #10foldするために各データ群を1回ずつテストデータに入力し残りをリサンプリングしてトレーニングデータに入力する
    for i in range(10):
        #テストデータを入れる箱を作成し，テストデータに各データ群を入れていく
        _test=divide_data[i]

        x_test=_test[:,1:21].astype(np.float32)
        y_test=_test[:,0].astype(np.int32)-1

        training_datapack = np.delete(divide_data,i,0)

        training_dataset=[]
        for k in training_datapack:
            for n in k:
                training_dataset.append(n.tolist())
        training_data_array=np.array(training_dataset)
        print training_data_array.shape

        training_data_list=training_data_array.tolist()
        random.shuffle(training_data_list)
        training_data_array = np.array(training_data_list)
        print training_data_array.shape
        print training_data_array[:, 1:21]
        
        #トレーニングデータに入れる
        x_train = np.array(training_data_array[:, 1:21]).astype(np.float32)
        y_train = np.array(training_data_array[:, 0]).astype(np.int32)-1

        #print x_train
        acc_loss=MLP.learning(x_train,y_train,x_test,y_test,n_fold=i,fname=fname)
        list_acc_loss.append(acc_loss)
    list_acc_loss=np.array(list_acc_loss)
#print list_acc_loss.shape
    #list_acc_loss=list_acc_loss.reshape(100,5)
    print np.average(list_acc_loss)
    #quit()
    #np.savetxt('{}_class_acc_loss.csv'.format(fname),list_acc_loss,delimiter=',')

#リサンプリングなし
def no_resampling_data2(divide_box,fname):
    list_acc_loss=[]
    divide_data=np.array(divide_box)
    nr.shuffle(divide_data)
    #print divide_data
    #print divide_data.shape
    #10foldするために各データ群を1回ずつテストデータに入力し残りをリサンプリングしてトレーニングデータに入力する
    for i in range(10):
        #テストデータを入れる箱を作成し，テストデータに各データ群を入れていく
        _test=divide_data[i]

        x_test=_test[:,1:21].astype(np.float32)
        y_test=_test[:,0].astype(np.int32)-1

        remaining_data=np.delete(divide_data,i,0)

        remain_data=[]
        for k in remaining_data:
            for n in k:
                remain_data.append(n.tolist())
        remained_data=np.array(remain_data)
        print remained_data.shape

        remained_data=remained_data.tolist()
        random.shuffle(remained_data)
        remained_data=np.array(remained_data)
        print remained_data.shape
        print remained_data[:, 1:21]
        #トレーニングデータに入れる
        x_train = np.array(remained_data[:, 1:21]).astype(np.float32)
        y_train = np.array(remained_data[:, 0]).astype(np.int32)-1
        #print x_train
        acc_loss=MLP.learning(x_train,y_train,x_test,y_test,n_fold=i,fname=fname)
        list_acc_loss.append(acc_loss)
    list_acc_loss=np.array(list_acc_loss)
#print list_acc_loss.shape
    #list_acc_loss=list_acc_loss.reshape(100,5)
    print np.average(list_acc_loss)
    #quit()
    #np.savetxt('{}_class_acc_loss.csv'.format(fname),list_acc_loss,delimiter=',')

if __name__=="__main__":
    data = np.loadtxt("tmp.csv", delimiter=",")
    print data
    data[:,0] += 1
    dataset = make_dataset(data)
    result = no_resampling_data(dataset, "test.csv")
    print result
    
    
    
    
