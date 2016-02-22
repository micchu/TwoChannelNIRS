# coding: utf-8

'''
Created on 2016/01/28
@author: misato

元ファイルは田村さん作
関数やパラメータを整理して整形

Update on 2016/02/01
回帰タイプNNを追加
'''


import sys
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F

from forward import forward_data, forward_data_regression
from test import test_nn_class, test_nn_regression

def train_nn_class(model, x_train, y_train, batchsize, epoch_num, test_flag = False, x_test = None, y_test = None, print_flag = False):
    """
    NNを学習させる
    Adamと呼ばれるパラメータの最適化手法を使用
    @param model: NNの構造モデルオブジェクト
    @param x_train: トレーニングデータの特徴量
    @param y_train: トレーニングデータの教師信号
    @param batchsize: 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    @param epoch_num: エポック数（1データセットの学習繰り返し数） 
    @keyword test_flag: テストデータの識別率を学習と同時並行して出力 
    """
    # 
    opts = optimizers.Adam()
    opts.setup(model)
    #optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    
    if print_flag:
        pace = 10 # 何epochごとに結果を出力するか
    if test_flag:
        pace = 10 # 何epochごとに結果を出力するか
        test_sample_size = len(x_test)
        
    #データ数
    sample_num = len(x_train)
        
    #学習ループ
    # 識別率とロス
    sum_accuracy = 0
    sum_loss = 0
    for epoch in xrange(1, epoch_num+1):        
        # サンプルの順番をランダムに並び替える
        perm = np.random.permutation(sample_num) # permutationはshuffleと違い、新しいリストを生成する
        
        # 1epochにおける識別率
        sum_accuracy_on_epoch = 0
        sum_loss_on_epoch = 0

        # データをバッチサイズごとに使って学習
        # 今回バッチサイズは1なので、サンプルサイズ数分学習
        for i in xrange(0, sample_num, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
    
            # 勾配(多分、偏微分分のgrad)を初期化
            opts.zero_grads()
            
            # 順伝播させて誤差と精度を算出
            loss, acc, output = forward_data(model, x_batch, y_batch, train=True)
            
            # 誤差逆伝播で勾配を計算
            loss.backward()
            
            # 勾配(gradを使って本物のweight)を更新
            opts.update()
            
            # 識別率，ロスの算出
            sum_loss_on_epoch += float(cuda.to_cpu(loss.data))
            sum_accuracy_on_epoch += float(cuda.to_cpu(acc.data)) # 田村さんに確認
        
        # 識別率とロスの積算
        accuracy_on_epoch = sum_accuracy_on_epoch / (sample_num / batchsize)
        loss_on_epoch = sum_loss_on_epoch / (sample_num / batchsize)
        sum_accuracy += accuracy_on_epoch
        sum_loss += loss_on_epoch
        
        if print_flag:
            if epoch % pace == 0:
                print 'epoch', epoch
                print 'train now: accuracy={}, loss={}'.format(accuracy_on_epoch, loss_on_epoch)
                #print 'train mean: loss={}, accuracy={}'.format(sum_loss / epoch, sum_accuracy / epoch)

        # テストデータでの誤差と、正解精度を表示。汎化性能を確認。 #########################
        if test_flag:
            if epoch % pace == 0:
                # テストデータでの誤差と、正解精度を表示
                accuracy, loss, result_class_list, result_loss_list, result_class_power_list = test_nn_class(model, x_test, y_test)
                print 'test: accuracy={}, loss={}'.format(accuracy, loss)

                
#                test_sum_accuracy = 0
#                test_sum_loss     = 0
#    
#                acc_t=np.zeros(((test_sample_size/batchsize), 5))
#                for i in xrange(0, test_sample_size, batchsize):
#                    x_batch = x_test[i:i+batchsize]
#                    y_batch = y_test[i:i+batchsize]
#                    
#                    # 順伝播させて誤差と精度を算出
#                    loss, acc, output= forward_data(model, x_batch, y_batch, train=False)
#                    acc_t[i][0]=loss.data
#                    acc_t[i][1]=acc.data
#                    acc_t[i][2]=y_batch #教師信号
#                    acc_t[i][3]=np.max(output.data)#value of max
#                    acc_t[i][4]=np.argmax(output.data)# 実際の出力結果
#                    
#                    test_sum_loss += float(cuda.to_cpu(loss.data))
#                    test_sum_accuracy += float(cuda.to_cpu(acc.data))
#                    
#                    print 'test: accuracy={}, loss={}'.format(test_sum_accuracy / (test_sample_size / batchsize), test_sum_loss / (test_sample_size / batchsize))

    return opts


def train_nn_regression(model, x_train, y_train, batchsize, epoch_num, test_flag = False, x_test = None, y_test = None, print_flag = False):
    """
    回帰タイプのNNを学習させる
    Adamと呼ばれるパラメータの最適化手法を使用
    @param model: NNの構造モデルオブジェクト
    @param x_train: トレーニングデータの特徴量
    @param y_train: トレーニングデータの教師信号
    @param batchsize: 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    @param epoch_num: エポック数（1データセットの学習繰り返し数） 
    @keyword test_flag: テストデータの識別率を学習と同時並行して出力 
    """
    # Adadeltaで学習
    opts = optimizers.AdaDelta() # class typeと違うよ
    opts.setup(model)
    
    if print_flag:
        pace = 10 # 何epochごとに結果を出力するか
    if test_flag:
        pace = 10 # 何epochごとに結果を出力するか
        test_sample_size = len(x_test)
        
    #データ数
    sample_num = len(x_train)
        
    #学習ループ
    # 識別率とロス
    sum_accuracy = 0
    sum_loss = 0
    for epoch in xrange(1, epoch_num+1):        
        # サンプルの順番をランダムに並び替える
        perm = np.random.permutation(sample_num) # permutationはshuffleと違い、新しいリストを生成する
        
        # 1epochにおける識別率
        sum_accuracy_on_epoch = 0
        sum_loss_on_epoch = 0

        # データをバッチサイズごとに使って学習
        # 今回バッチサイズは1なので、サンプルサイズ数分学習
        for i in xrange(0, sample_num, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
    
            # 勾配(多分、偏微分分のgrad)を初期化
            opts.zero_grads()
            
            # 順伝播させて誤差と予測結果を算出
            loss, predicted = forward_data_regression(model, x_batch, y_batch)
            
            # 誤差逆伝播で勾配を計算
            loss.backward()
            
            # 勾配(gradを使って本物のweight)を更新
            opts.update()
            
            # ロスの算出
            sum_loss_on_epoch += float(cuda.to_cpu(loss.data))
        
        # 識別率とロスの積算
        loss_on_epoch = sum_loss_on_epoch / (sample_num / batchsize)
        sum_loss += loss_on_epoch
        
        if print_flag:
            if epoch % pace == 0:
                print 'epoch', epoch
                print 'train now: loss={}'.format(loss_on_epoch)
                #print 'train mean: loss={}, accuracy={}'.format(sum_loss / epoch, sum_accuracy / epoch)

        # テストデータでの誤差と、正解精度を表示。汎化性能を確認。 #########################
        if test_flag:
            if epoch % pace == 0:
                predicted_value_list, loss_list = test_nn_regression(model, x_test, y_test)
                print 'test: loss={}, teach={}, predicted={}'.format(np.average(loss_list), y_test, predicted_value_list)
                
#                test_sum_accuracy = 0
#                test_sum_loss     = 0
#    
#                acc_t=np.zeros(((test_sample_size/batchsize), 5))
#                for i in xrange(0, test_sample_size, batchsize):
#                    x_batch = x_test[i:i+batchsize]
#                    y_batch = y_test[i:i+batchsize]
#                    
#                    # 順伝播させて誤差と精度を算出
#                    loss, predicted = forward_data_reg(model, x_batch, y_batch, train=False)
#                    acc_t[i][0]=float(cuda.to_cpu(loss.data))
#                    acc_t[i][1]=float(cuda.to_cpu(predicted.data))
#                    acc_t[i][2]=float(cuda.to_cpu(y_batch)) #教師信号
#                    
#                    test_sum_loss += float(cuda.to_cpu(loss.data))
#                    
#                # テストデータでの誤差と、正解精度を表示
#                print 'test: loss={}'.format(test_sum_loss / (test_sample_size / batchsize))

    return opts

if __name__=="__main__":
    #データセットをロードしてランダムに並べ替え
    from sklearn.datasets import fetch_mldata
    from test import test_nn_class
    import random
    
    if True: # クラス文類タイプ
        print 'fetch MNIST dataset'
        mnist = fetch_mldata('MNIST original')
        # mnist.data : 70,000件の784次元ベクトルデータ
        mnist.data   = mnist.data.astype(np.float32)
        mnist.data  /= 255     # 0-1のデータに変換
        
        # mnist.target : 正解データ（教師データ）
        mnist.target = mnist.target.astype(np.int32)
    
        N = 1000
        N_test = 200
        Robj = random.Random()
        Robj.seed(100)
        index_list = range(70000)
        Robj.shuffle(index_list)
        
        x_train = mnist.data[index_list[:N]]
        x_test = mnist.data[index_list[N:N+N_test]]
        y_train = mnist.target[index_list[:N]]
        y_test = mnist.target[index_list[N:N+N_test]]
    #    N = 60000
    #    x_train, x_test = np.split(mnist.data,   [N])
    #    y_train, y_test = np.split(mnist.target, [N])
#        N_test = y_test.size
        
        n_units = 1000
        
        model = FunctionSet(l1=F.Linear(784, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, 10))
        
        train_nn_class(model, x_train, y_train, 100, 20, print_flag=True)
        ret = test_nn_class(model, x_test, y_test)
        print "Final Result"
        print "acc", ret[0]
        print "loss", ret[1]
        print "res", ret[2]
        print "power", ret[3]
        #numpy配列のtypeをfloat32またはint32にしないといけない
        #トレーニングデータとテストデータに分ける
    #    x_train = np.array(dataset[0:460, :30], dtype=np.float32)
    #    x_test = np.array(dataset[460:, :30], dtype=np.float32)
    #    y_train = np.array(dataset[0:460, -1], dtype=np.int32)
    #    y_test = np.array(dataset[460:, -1], dtype=np.int32)
    
    # Regressionタイプ 
    if False:
        pass