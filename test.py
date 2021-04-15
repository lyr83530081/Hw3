# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:12:07 2021

@author: user
"""

import pickle
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import time
#import common
from collections import OrderedDict
import pickle

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
 
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def softmax(x):
    c=np.max(x,axis=1,keepdims=True)
    exp_x=np.exp(x-c)
    sum_exp_x=np.sum(exp_x,axis=1,keepdims=True)
    return exp_x/sum_exp_x

def cross_entropy_error(y,t):
    delta=1e-7
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+delta))/batch_size


class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    def backward(self,dout):
        dout[self.mask]=0
        return dout
  
class Sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        self.out=1/(np.exp(-x)+1)
        return self.out
    def backward(self,dout):
        return dout*self.out*(1-self.out)
  
class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.original_x_shape = None
        self.dW=None
        self.db=None
    def forward(self,x):
        self.original_x_shape = x.shape
        x=x.reshape(x.shape[0],-1)
        self.x=x
        return np.dot(x,self.W)+self.b
    def backward(self,dout):
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        dx=np.dot(dout,self.W.T)
        return dx.reshape(*self.original_x_shape)

class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batch_size=self.y.shape[0]
        return (self.y-self.t)/batch_size
    
class BatchNormalization:
    def __init__(self,gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.x_=None
        self.dgamma = None
        self.dbeta = None
    def forward(self,x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
          
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            sample_mean = x.mean(axis=0)
            sample_var = x.var(axis=0)
            self.batch_size = x.shape[0]
            self.var_plus_eps=sample_var+10e-7
            self.x_ = (x - sample_mean) / np.sqrt(sample_var + 10e-7)
            
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * sample_var            
        else:
            xc = x - self.running_mean
            self.x_ = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * self.x_ + self.beta 
        return out.reshape(*self.input_shape)

    def backward(self,dout):
        # calculate gradients
        self.dgamma = np.sum(self.x_ * dout, axis=0)
        self.dbeta = np.sum(dout, axis=0)
      
        dx_ = np.matmul(np.ones((self.batch_size,1)), self.gamma.reshape((1, -1))) * dout
        dx = self.batch_size * dx_ - np.sum(dx_, axis=0) - self.x_ * np.sum(dx_ * self.x_, axis=0)
        dx *= (1.0/self.batch_size) / np.sqrt(self.var_plus_eps)
      
        return dx
  
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

#        self.x = x
        self.xshape=x.shape
        self.col = col
        self.col_W = col_W

        return out
 
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
#        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        dx = col2im(dcol, self.xshape, FH, FW, self.stride, self.pad)

        return dx
  
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

#        self.x = x
        self.xshape=x.shape
        self.arg_max = arg_max

        return out
 
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
#        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        dx = col2im(dcol, self.xshape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


class Lenet5:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num1':6, 'filter_size1':3,'filter_num2':16, 'filter_size2':3, 'pad':1, 'stride':1},
                 hidden_size1=120,hidden_size2=84, output_size=10, weight_init_std=0.01):
        filter_num1 = conv_param['filter_num1']
        filter_size1 = conv_param['filter_size1']
        filter_num2 = conv_param['filter_num2']
        filter_size2 = conv_param['filter_size2']
        filter_num3 = conv_param['filter_num3']
        filter_size3 = conv_param['filter_size3']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # conv_output_size1 = (input_size - filter_size1 + 2*filter_pad) / filter_stride + 1
        # pool_output_size1 = (conv_output_size1 - 2) / 2 + 1
        # conv_output_size2 = (pool_output_size1 - filter_size2 + 2*filter_pad) / filter_stride + 1
        # pool_output_size2 = int((((conv_output_size2 - 2) / 2 + 1)**2)*filter_num2)
        # conv_output_size3 = (pool_output_size2 - filter_size3 + 2*filter_pad) / filter_stride + 1
        #pool_output_size3 = int((((conv_output_size3 - 2) / 2 + 1)**2)*filter_num3)
        pool_output_size3 = 2592
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2)
        self.params['b2'] = np.zeros(filter_num2)
        self.params['W_add'] = weight_init_std * \
                            np.random.randn(filter_num3, filter_num2, filter_size3, filter_size3)
        self.params['b_add'] = np.zeros(filter_num3)
        
        self.params['W3'] = weight_init_std * \
                            np.random.randn(pool_output_size3, hidden_size1)
        self.params['b3'] = np.zeros(hidden_size1)
        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size1, hidden_size2)
        self.params['b4'] = np.zeros(hidden_size2)
        self.params['W5'] = weight_init_std * \
                            np.random.randn(hidden_size2, output_size)
        self.params['b5'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        ### the added conv 
        self.layers['Conv3'] = Convolution(self.params['W_add'], self.params['b_add'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        ###
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithCrossEntropy()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        #从独热编码转回数字编码
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W_add'], grads['b_add'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W5'], grads['b5'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads
        
    #只保留权重信息，不包含网络模型
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2' , 'Conv3' ,'Affine1', 'Affine2','Affine3']):
            if i == 2:
                self.layers[key].W = self.params['W_add']
                self.layers[key].b = self.params['b_add']
            elif i>2:
                self.layers[key].W = self.params['W' + str(i)]
                self.layers[key].b = self.params['b' + str(i)]
            else:
                self.layers[key].W = self.params['W' + str(i+1)]
                self.layers[key].b = self.params['b' + str(i+1)]

            

def data_loader(data,batch_sz,batch_num,Y_list):
    data_num = len(data)
    l1,l2=data_num//batch_sz, data_num%batch_sz
    #start_time = time.time()
    if batch_num == l1+1:
        Batch = []
        batch_lable = Y_list[data_num-l2:data_num,:]
        for i in range(data_num-l2,data_num):
            img_path = data[i].split()
            img = cv2.imread(img_path[0])
            img_re = cv2.resize(img, (72, 72), interpolation=cv2.INTER_AREA)
            Batch.append(img_re)
    else:
        
        #end_time = time.time()
        #print(start_time-end_time)
        batch_lable = Y_list[(batch_num-1)*batch_sz:batch_num*batch_sz,:]
        Batch = []
        for i in range((batch_num-1)*batch_sz,batch_num*batch_sz):
            #start_time = time.time()
            img_path = data[i].split()
            #end_time = time.time()
            #print(start_time-end_time)
            img = cv2.imread(img_path[0])
            img_re = cv2.resize(img, (72, 72), interpolation=cv2.INTER_AREA)
            Batch.append(img_re)
    return Batch,batch_lable


def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

output_size=50
with open('train.txt', 'r') as f:
    index = f.readlines() 
    state=np.random.get_state()
    np.random.shuffle(index)
    lable_list = [0]*len(index)
    k=0
    for line in index:
        img_lable = line.split()
        lable_list[k] = int(img_lable[1])
        k+=1
    lable_list = np.array(lable_list)
    
with open('val.txt', 'r') as f:
    index_val = f.readlines() 
    #state=np.random.get_state()
    np.random.shuffle(index_val)
    lable_list_val = [0]*len(index_val)
    k=0
    for line in index_val:
        img_lable = line.split()
        lable_list_val[k] = int(img_lable[1])
        k+=1
    lable_list_val = np.array(lable_list_val)
Y_list = MakeOneHot(lable_list, output_size)
Y_list_val = MakeOneHot(lable_list_val, output_size)

max_epochs = 10
batch_size = 140
network = Lenet5(input_dim=(3,72,72), 
                 conv_param={'filter_num1':6, 'filter_size1':3,'filter_num2':16, 'filter_size2':3, 'filter_num3':32, 'filter_size3':3 , 'pad':1, 'stride':1},
                 hidden_size1=3000,hidden_size2=500, output_size=50, weight_init_std=0.01)

network.load_params()
loss_train = []
loss_val = []
acc_list = []
acc_list_val = []


correct_val = 0
k=1
loss_v = 0

while(True):
    batch_val,batch_lable_val = data_loader(index_val,batch_size,k,Y_list_val)
    batch_val = np.array(batch_val).transpose((0,3,1,2))/255
    y = network.predict(batch_val)
    loss = network.last_layer.forward(y, batch_lable_val)
    result = np.argmax(y, axis=1) - np.argmax(batch_lable_val, axis=1)
    result = list(result)
    correct_val += result.count(0)
    loss_v+=loss
    print(k)
    k+=1
    if k==len(index_val)//batch_size+2:
        break
loss_v = loss_v/(k-1)  
loss_val.append(loss_v)
print(" Loss_val= {:.5},acc_val = {:.5}%".format(loss_v,100*correct_val/len(index_val)))
acc_list_val.append(100*correct_val/len(index_val))