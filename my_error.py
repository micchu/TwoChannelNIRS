#coding:utf-8

'''
Created on 2016/01/28

@author: misato
'''

class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

