import torch
import pdb
import numpy as np
import torch.nn.functional as F

def ToOnehot(label, num_classes):
    temp = torch.eye(num_classes)
    return temp[label]

def ToOnehots(labels,num_classes):
    onehots = torch.zeros(size=[labels.size(0),num_classes])
    # pdb.set_trace()
    for i in range(labels.size(0)):
        onehots[i] = ToOnehot(labels[i],num_classes)
    return onehots

def MySoftmax(x, axis=1):#采用矩阵运算，尽量不要用for循环
    x_max,_ = x.max(axis=axis, keepdims=True)
    x = x - x_max
    y = torch.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def Onehot_test():
    label = torch.tensor(5)
    labels=torch.tensor([0,5,6])
    a = ToOnehot(label, 10)
    b = ToOnehots(labels,10)
    print(a)
    print(b)

def Softmax_test():
    label = np.array([[0.,0.,1.], [0., 1., 0.]])#直接输入onehot测试
    label1 = torch.from_numpy(label)

    a = MySoftmax(label1)
    b = F.softmax(label1)

    print(a)
    print(b)
if __name__=="__main__":
    # Onehot_test()
    Softmax_test()
