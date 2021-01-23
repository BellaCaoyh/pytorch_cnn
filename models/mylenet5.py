from utils.layers import *
import sys,os
sys.path.append(os.pardir)
import torch
import pickle
import numpy as np



class MyLenet5:
    """
    网络结构：
    c1: conv1 -- maxpool -- relu
    c2: conv2 -- maxpool -- relu
    c3: conv3 -- relu
    f4: Linear/Affine -- sigmoid
    f5: Linear/Affine -- softmax
    """
    def __init__(self, input_dim=(3,32,32),
                 conv1_param = {'filter_num':6,'filter_size':5,'stride':1,'pad':0},
                 conv2_param = {'filter_num':16,'filter_size':5,'stride':1,'pad':0},
                 conv3_param = {'filter_num':120,'filter_size':5,'stride':1,'pad':0},
                 hidden_size=84,output_sie=10):

        '''初始化权重'''
        pre_node_nums = np.array([1*5*5,6*5*5,16*5*5,120*5*5,hidden_size])
        weight_init_scale = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv1_param,conv2_param,conv3_param]):
            self.params['W' + str(idx+1)] = weight_init_scale[idx] * np.random.randn(conv_param['filter_num'],pre_channel_num,
                                                                                     conv_param['filter_size'],conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.params['W4'] = weight_init_scale[3] * np.random.randn(120,hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = weight_init_scale[4] * np.random.randn(hidden_size,output_sie)
        self.params['b5'] = np.zeros(output_sie)

        # 生成层======================
        self.layers = []
        #C1
        self.layers.append(Convolution(self.params['W1'],self.params['b1'],
                                       conv1_param['stride'],conv1_param['pad']))
        self.layers.append(Pooling(pool_h=2,pool_w=2,stride=2))
        self.layers.append(Relu())
        #C2
        self.layers.append(Convolution(self.params['W2'],self.params['b2'],
                                       conv2_param['stride'],conv2_param['pad']))
        self.layers.append(Pooling(pool_h=2,pool_w=2,stride=2))
        self.layers.append(Relu())
        #C3
        self.layers.append(Convolution(self.params['W3'],self.params['b3'],
                                       conv3_param['stride'],conv3_param['pad']))
        self.layers.append(Relu())
        #f4
        self.layers.append(Affine(self.params['W4'],self.params['b4'],))
        self.layers.append(Sigmoid())
        #f5
        self.layers.append(Affine(self.params['W5'],self.params['b5']))

        self.last_layer = SoftmaxWithLoss()

    def predict(self,x,train_flg=False):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y,t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: #ndim 返回数组的维度
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size: (i+1)*batch_size]
            tt = t[i * batch_size: (i+1)*batch_size]

            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x,t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)



        temp_layers = self.layers.copy()
        temp_layers.reverse() #list 对象反向排列

        # pdb.set_trace()
        for layer in temp_layers:
            dout = layer.backward(dout)

        #Setting
        grads = {}
        for i, layer_idx in enumerate((0,3,6,8,10)):#include Conv&Affline
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key,val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key,val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0,3,6,8,10)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]







# if __name__=='__main__':
#     lenet5_test()











































