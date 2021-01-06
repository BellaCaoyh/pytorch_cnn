import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("..")
from utils import ToOnehot,MySoftmax
import pdb

class MyCrossEntropy(nn.Module):
    def __init__(self,num_classes=10):
        super(MyCrossEntropy,self).__init__()
        self.num_classes = num_classes

    def forward(self,y,label):
        length = y.size(0)
        delta = 1e-7
        softmax_out = MySoftmax(y)
        log_out = torch.log(softmax_out + delta)
        # pdb.set_trace()
        loss = - torch.sum(label * log_out)
        loss = loss / length
        # loss = -torch.sum(label * torch.log(y + delta))/length
        return loss



def test():
    label1 = np.array([2])
    label1 = torch.from_numpy(label1)
    y = torch.tensor([[2.007,9.550,6.024]])

    MCE = MyCrossEntropy(num_classes=3)
    a = MCE(y,label1)

    CE = nn.CrossEntropyLoss()
    b = CE(y,label1)

    print(a,b)

if __name__ == '__main__':
    # test()
    # y = [0,1,2,3]
    # a = MySoftmax(y)
    # print(a)
    # print(sum(a))

    y = torch.randn(3,4)
    # pdb.set_trace()
    # out = MySoftmax_2(y)
    # print(out.sum(axis=1))
    out1 = softmax(y)
    out2 = F.softmax(y)
    print(out1)
    print(out2)









