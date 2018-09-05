# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 17:10:43 2018

@author: huanghu
"""

import torch
from NeuralNetwork import NeuralNetwork as ne
class AND:
    def __init__(self, data):
        self.gate=ne([2,1])
        self.gate.getlayer(0)[0]=1
        self.gate.getlayer(0)[1]=1
        self.data=data
    def __call__(self,data):
        self.data=data
    def forward(self):
        self.data=self.gate.forward(self.data)
        result=[]
        n=len(self.data)[0]
        for i in range(n):
            if self.data[i]>0.8:
                result.append(True)
            else:
                result.append(False)
        return result
        
class NOT:
    def __init__(self,data):
        self.data=data
        self.gate=ne([1,1])
        self.gate.getlayer(0)[0]=1
    def __call__(self,data):
        self.data=data
    def forward(self):
        self.data=self.gate.forward(self.data)
        result=[]
        n=len(self.data)[0]
        for i in range(n):
            if self.data[i]>0.6:
                result.append(True)
            else:
                result.append(False)
        return result
        
class XOR:
    def __init__(self,data):
        self.gate=ne([2,1])
        self.gate.getlayer(0)[0]=1
        self.gate.getlayer(0)[1]=1
        self.data=data
    def __call__(self,data):
        self.data=data
    def forward(self):
        self.data=self.gate.forward(self.data)
        result=[]
        n=len(self.data)[0]
        for i in range(n):
            if self.data[i]>0.6 and self.data[i]<0.8:
                result.append(True)
            else:
                result.append(False)
        return result
        
class OR:
    def __init__(self, data):
        self.gate=ne([2,1])
        self.gate.getlayer(0)[0]=1
        self.gate.getlayer(0)[1]=1
        self.data=data
    def __call__(self,data):
        self.data=data
    def forward(self):
        self.data=self.gate.forward(self.data)
        result=[]
        n=len(self.data)[0]
        for i in range(n):
            if self.data[i]>0.6:
                result.append(True)
            else:
                result.append(False)
        return result

        
        
a=XOR(torch.DoubleTensor([[1,0]]))
a=a.forward()
print (a)