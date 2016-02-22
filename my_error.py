#coding:utf-8

'''
Created on 2016/01/28

@author: misato
'''

class MyError(Exception):
    def __init__(self, value, value2):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
if __name__=="__main__":
    try:
        k = 12
        print "test" + k
    except Exception, e:
        MyError("String Error", e.msg)

