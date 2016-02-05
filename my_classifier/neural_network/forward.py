#coding:utf-8

'''
Created on 2016/01/28
@author: misato

Update on 2016/02/01
回帰タイプNNを追加
'''

from chainer import Variable
import chainer.functions as F

def forward_data(model, x_data, y_data, train=True):
    """
    @param model: NNの構造モデルオブジェクト
    @param x_data: 特徴量
    @param y_data: 教師信号
    """
    #データをnumpy配列からChainerのVariableという型(クラス)のオブジェクトに変換して使わないといけない
    x, t = Variable(x_data), Variable(y_data)

    #ドロップアウトでオーバーフィッティングを防止
    h1 = F.dropout(F.relu(model.l1(x)), ratio=0.4, train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), ratio=0.5, train=train)

    y  = model.l3(h2)
    
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出
    
    #F.accuracy()は識別率を算出
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t), F.softmax(y)


def forward_data_regression(model, x_data, y_data, train=True):
    """
    回帰計算用のNNの伝搬計算
    @param model: NNの構造モデルオブジェクト
    @param x_data: 特徴量
    @param y_data: 教師信号
    @return: 誤差と予測結果を返す
    """
    #データをnumpy配列からChainerのVariableという型(クラス)のオブジェクトに変換して使わないといけない
    x, t = Variable(x_data), Variable(y_data)

    #ドロップアウトでオーバーフィッティングを防止
    h1 = F.dropout(F.relu(model.l1(x)), ratio=0.4, train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), ratio=0.5, train=train)

    y  = model.l3(h2)
    
    # 回帰なので誤差関数として最小二乗誤差を利用
    # F.mean_squared_error()は誤差
    # yは出力結果
    return F.mean_squared_error(y, t), y


