# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:14:22 2018

@author: huanghu
"""

import torch
import math
class NeuralNetwork(object):
    def __init__(self,layer_lists):
        self.layer_lists=layer_lists
        self.theta=[]
        for i in range(len(layer_lists)-1):
            self.theta.append(torch.DoubleTensor(layer_lists[i],layer_lists[i+1]))
            self.theta[i]=torch.normal(mean=torch.zeros(layer_lists[i],layer_lists[i+1],dtype=torch.double),std=math.sqrt(1/layer_lists[i]),out=self.theta[i])
            
    def getlayer(self,layer):
        return self.theta[layer]
    def forward(self,data):
        n=len(self.layer_lists)
        out=data
        for i in range (n-1):
            a=self.getlayer(i)
            out=torch.mm(out,a)
            out=torch.sigmoid(out)
        return out
    
#test the class
