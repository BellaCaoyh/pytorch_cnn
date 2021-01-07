import sys
sys.path.append("..")
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ToOnehot,MySoftmax


class MyCrossEntropy(nn.Module):#自定义损失函数
    def __init__(self,num_classes=10):
        super(MyCrossEntropy,self).__init__()
        self.num_classes = num_classes

    def forward(self, y, label):#输入的标签值应为onehot值，pytorch自带CEL损失函数则不需要
        length = y.size(0)
        delta = 1e-7
        y = MySoftmax(y)
        loss = -torch.sum(label * torch.log(y + delta))/length
        return loss


def CEL_test():
    label1 = np.array([2])
    label1 = torch.from_numpy(label1)
    y = torch.tensor([[2.007,9.550,6.024]])

    label2 = ToOnehot(label1,3)
    MCE = MyCrossEntropy(num_classes=3)
    a = MCE(y,label2)

    CE = nn.CrossEntropyLoss()
    b = CE(y,label1)

    print(a)
    print(b)

if __name__ == '__main__':
    CEL_test()










