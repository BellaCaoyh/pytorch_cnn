import torch
import pdb
import numpy as np
import torch.nn.functional as F
import pdb


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

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

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Args:
        input_data: 由（N, C, H, W）__(数据量，通道数，图片高，图片宽)的4维数组构成的输入数据
        filter_h: 滤波器的高
        filter_w: 滤波器的宽
        stride: 步长，默认为1
        pad: 填充，默认为0

    Returns: col 2维数组

    """
    N, C, H, W = input_data.shape #100,1,28,28
    out_h = (H + 2*pad - filter_h)//stride + 1 #28
    out_w = (W + 2*pad - filter_w)//stride + 1 #28
    # pdb.set_trace()

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')#img.shape = 100,1,30,30
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) # N, C, filter_h, filter_w, out_h, out_w

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # pdb.set_trace()
    # col.shape: 100,1,3,3,28,28 N, C, filter_h, filter_w, out_h, out_w --> N, out_h, out_w, C, filter_h, filter_w,
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) #N, out_h, out_w, C, filter_h, filter_w --> N*out_h*out_w, C*filter_h*filter_w
    return col # N*out_h*out_w, C*filter_h*filter_w


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Args:
        col: 2维数组 N*out_h*out_w, C*filter_h*filter_w
        input_shape: 由（N, C, H, W）__(数据量，通道数，图片高，图片宽)的4维数组构成的输入数据
        filter_h: 滤波器的高
        filter_w: 滤波器的宽
        stride: 步长，默认为1
        pad: 填充，默认为0

    Returns:img

    """
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



# if __name__=="__main__":
#     Onehot_test()
    # Softmax_test()
