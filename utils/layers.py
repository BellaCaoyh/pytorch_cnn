import numpy as np
from utils.utils import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx



class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self,x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0],-1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        #
        self.x = None
        self.col = None
        self.col_W = None

        #dw & db
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape #W[FN,C,FH,FW]  FN：滤波器个数 C：通道数 FH：滤波器高 FW：滤波器的宽
        N, C, H, W = x.shape #x[N,C,H,W] N：输入图像的个数 C:通道数 H：图像的高 W：图像的宽

        out_h = 1 + int((H + 2*self.pad - FH) / self.stride) #输出h
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride) #输出w

        col = im2col(x, FH, FW, self.stride, self.pad) #x-->col col:[N*out_h*out_w, C*FH*FW]
        col_W = self.W.reshape(FN,-1).T #W-->col_W col_W:[FN, C*FH*FW] --> col_W:[C*FH*FW, FN]

        out = np.dot(col,col_W) #out=col*col_W  out:[N*out_h*out_w, FN]
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) #out:[N,out_h,out_w, FN] --> [N, FN, out_h, out_w]

        self.x = x #x:[N,C,H,W]
        self.col = col #col:[N*out_h*out_w, C*FH*FW]
        self.col_W = col_W #col_W:[C*FH*FW, FN]

        return out

    def backward(self, dout):
        FN,C,FH,FW = self.W.shape
        dout =  dout.transpose(0,2,3,1).reshape(-1,FN) #dout:[N,FN,out_h,out_w] --> [N,out_h,out_w,FN] --> [N*out_h*out_w,FN]

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout) #col:[N*out_h*out_w, C*FH*FW] --> [C*FH*FW,N*out_h*out_w]  dW:[N*out_h*out_w,FN]
        self.dW = self.dW.transpose(1,0).reshape(FN,C,FH,FW) #dW:[FN,C,FH,FW]

        dcol = np.dot(dout, self.col_W.T) #dcol:[N*out_h*out_w,C*FH*FW]
        dx = col2im(dcol,self.x.shape, FH,FW,self.stride,self.pad) #dx:[N,C,out_h,out_w]

        return dx


class Pooling:
    def __init__(self,pool_h,pool_w, stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N,C,H,W = x.shape

        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1) #获取每一行最大值的索引值
        out = np.max(col,axis=1) #获取每一行最大值
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax =np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # pdb.set_trace()

        dmax = dmax.reshape(dout.shape + (pool_size,)) #dout.shape[2,5,5,16] pool_size=4
        # dmax = dmax.reshape((dout.size,pool_size))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] *dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,self.pool_w,self.stride,self.pad)

        return dx





















