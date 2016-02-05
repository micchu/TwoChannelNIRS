
# coding: utf-8

# In[4]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys
#from bokeh.server import forwarder
#from docutils.utils.punctuation_chars import delimiters


# In[63]:
'''
#データセットをロードしてランダムに並べ替え
filename = '../Desktop/wdbc.npy'
dataset = np.load(filename)
np.random.shuffle(dataset)
dataset


# In[61]:

#numpy配列のtypeをfloat32またはint32にしないといけない
#トレーニングデータとテストデータに分ける
x_train = np.array(dataset[0:460, :30], dtype=np.float32)
x_test = np.array(dataset[460:, :30], dtype=np.float32)
y_train = np.array(dataset[0:460, -1], dtype=np.int32)
y_test = np.array(dataset[460:, -1], dtype=np.int32)

'''
# In[56]:

# ニューラルネットの構造
def forward(x_data, y_data, train=True):
    #データをnumpy配列からChainerのVariableという型(クラス)のオブジェクトに変換して使わないといけない
    x, t = Variable(x_data), Variable(y_data)
    #ドロップアウトでオーバーフィッティングを防止
    h1 = F.dropout(F.relu(model.l1(x)), ratio=0.4, train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), ratio=0.5, train=train)
    #h1 = F.sigmoid(model.l1(x))
    #h2 = F.sigmoid(model.l2(h1))
    y  = model.l3(h2)
    
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出
    
    #F.accuracy()は識別率を算出
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t), F.softmax(y)

#5class分類
n_units1   = 30
n_units2   = 30
model = FunctionSet(l1=F.Linear(20, n_units1),
                        l2=F.Linear(n_units1, n_units2),
                        l3=F.Linear(n_units2, 5))
'''
#3class分類
n_units1   = 30
n_units2   = 30
model = FunctionSet(l1=F.Linear(20, n_units1),
                        l2=F.Linear(n_units1, n_units2),
                        l3=F.Linear(n_units2, 3))

'''
# In[57]:
def learning(x_train,y_train,x_test,y_test,n_fold,fname):
    # 多層パーセプトロンモデルの設定
    # 入力 30次元、出力 2次元
    #F.Linear（層の入力数，層の出力数）
    # 中間層の数
    
    # Adamと呼ばれるパラメータの最適化手法らしい
    #オンライン学習器の一種で勾配の期待値を用いて重みの更新を行う？
    #この方法だと学習率を下げる必要がない
    optimizer = optimizers.Adam()
    #optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    
    train_loss = []
    train_acc  = []
    test_loss = []
    test_acc  = []
    
    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batchsize = 1
    
    # 学習させる回数
    n_epoch   = 200
    
    '''
    #既存手法
    #データ数
    N = 90
    N_test = 10
    '''
    
    #データ数
    N = 90
    N_test = 10
    
    #printする間隔
    pace = 10
    
    #学習ループ
    for epoch in xrange(1, n_epoch+1):
        sum_acc = 0
        # training
        # N個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        # 0〜Nまでのデータをバッチサイズごとに使って学習
        for i in xrange(0, N, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
    
            # 勾配を初期化
            optimizer.zero_grads()
            # 順伝播させて誤差と精度を算出
            loss, acc,output = forward(x_batch, y_batch)
            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()
    
            #train_loss.append(loss.data)
            #train_acc.append(acc.data)
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
    
        # 訓練データの誤差と、正解精度を表示
        if epoch % pace == 0:
            print 'epoch', epoch
            print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)
    
        # evaluation
        # テストデータで誤差と、正解精度を算出し汎化性能を確認
        sum_accuracy = 0
        sum_loss     = 0
        
        if epoch==200:
            acc_t=np.zeros((10,5))
            for i in xrange(0, N_test, batchsize):
                x_batch = x_test[i:i+batchsize]
                y_batch = y_test[i:i+batchsize]
                
                # 順伝播させて誤差と精度を算出
                loss, acc ,output= forward(x_batch, y_batch, train=False)
                acc_t[i][0]=loss.data
                acc_t[i][1]=acc.data
                acc_t[i][2]=y_batch #教師信号
                acc_t[i][3]=np.max(output.data)#value of max
                acc_t[i][4]=np.argmax(output.data)# 実際の出力結果
                
                #test_loss.append(loss.data)
                #test_acc.append(acc.data)
                sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
        else:
            for i in xrange(0, N_test, batchsize):
                x_batch = x_test[i:i+batchsize]
                y_batch = y_test[i:i+batchsize]
                
                # 順伝播させて誤差と精度を算出
                loss, acc ,output= forward(x_batch, y_batch, train=False)
                #test_loss.append(loss.data)
                #test_acc.append(acc.data)
                sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
        # テストデータでの誤差と、正解精度を表示
        if epoch % pace == 0:
            print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)
    
        sum_acc += sum_accuracy
    print sum_acc / 10
    #print acc_t
    return acc_t
    np.savetxt('{}_class_acc_loss_Re_{}.csv'.format(fname,n_fold),acc_t,delimiter=',')
    # 精度と誤差をグラフ描画
    plt.close()
    plt.figure(figsize=(12,10))
    plt.plot(range(len(train_acc)), train_acc, 'b')
    plt.plot(range(len(test_acc)), test_acc, 'r')
    plt.legend(["train_acc","test_acc"],loc=4)
    plt.ylim([0.5,1.1])
    plt.title("Accuracy of digit recognition.")
    plt.plot()
    plt.savefig("plot_fig")

# In[ ]:



