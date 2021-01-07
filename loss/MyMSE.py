import sys
sys.path.append("..")
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ToOnehot,MySoftmax


class MyMSE(nn.Module):
    def __init__(self):
        super(MyMSE,self).__init__()

    def forward(self,y,label):##输入的标签值应为onehot值
        y = MySoftmax(y)
        # y = F.softmax(y,dim=1)
        # label = ToOnehot(label,3)
        loss = torch.sum((y - label)**2) / (y.size(0) * y.size(1))
        return loss

def MSE_test():

    label = np.array([[0.,0.,1.], [0., 1., 0.]])#直接输入onehot测试
    label1 = torch.from_numpy(label)
    y = torch.tensor([[2.007,9.550,6.024], [1.005, 3.567, 2.897]])

    CE = nn.MSELoss(reduction='mean')#pytorch自带的损失函数对于y在输入前应该先通过softmax
    y1 = MySoftmax(y)
    a = CE(y1,label1)
    print('Pytorch_MSE:\n',a)

    MS = MyMSE()
    b = MS(y,label1)
    print('MyMSE: \n',b)

if __name__ == '__main__':
    MSE_test()

