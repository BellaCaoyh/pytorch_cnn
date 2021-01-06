import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("..")
from utils import ToOnehot,MySoftmax
import pdb

class MyMSE(nn.Module):
    def __init__(self):
        super(MyMSE,self).__init__()

    def forward(self,y,label):
        y = MySoftmax(y)
        # y = F.softmax(y,dim=1)
        # label = ToOnehot(label,3)
        loss = torch.sum((y - label)**2) / 6
        return loss

def test():
    # label0 = 2
    label = np.array([[0.,0.,1.], [0., 1., 0.]])
    label1 = torch.from_numpy(label)
    y = torch.tensor([[2.007,9.550,6.024], [1.005, 3.567, 2.897]])

    label2 = torch.from_numpy(label1)

    CE = nn.MSELoss(reduction='mean')
    y2 = MySoftmax(y)
    b = CE(y2,label2)
    print(b)

    MS = MyMSE()
    a = MS(y,label1)
    print(a)

if __name__ == '__main__':
    test()

