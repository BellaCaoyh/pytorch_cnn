import torch
import pdb
import numpy as np

# def ToOnehot(labels,num_classes):
#     temp = torch.eyes(len(labels),num_classes)
#     for i in range(len(labels)):
#         temp[i,labels[i]] = 1
#     return temp

def ToOnehot(label, num_classes):
    temp = torch.eye(num_classes)
    return temp[label]

def ToOnehots(labels,num_classes):
    onehots = torch.zeros(size=[labels.size(0),num_classes])
    # pdb.set_trace()
    for i in range(labels.size(0)):
        onehots[i] = ToOnehot(labels[i],num_classes)
    return onehots

def MySoftmax(x, axis=1):
    x_max,_ = x.max(axis=axis, keepdims=True)
    x = x - x_max
    y = torch.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def test():
    labels=torch.tensor([0,5,6])
    label = torch.tensor(5)
    a = ToOnehot(label, 10)
    b = ToOnehots(labels,10)
    # pdb.set_trace()
    print(a)
    print(b)

if __name__=="__main__":
    test()
