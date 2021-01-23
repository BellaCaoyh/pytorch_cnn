import torch
import pdb
import argparse
import random
from models import *
import numpy as np
from dataset.dataset import Cifar10Dataset
from mmcv import Config
from torch.utils.data import DataLoader
from log.logger import Logger
from utils.get_net import get_network
from utils.utils import ToOnehots
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    parser.add_argument('--net', '-n', type=str, required=True, help='input which model to use')
    parser.add_argument('--pretrain', '-p', action='store_true', help='Location pretrain data')
    parser.add_argument('--gpuid', '-g', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    return args

def dataLoad (cfg):
    test_data = Cifar10Dataset(txt = cfg.PARA.cifar10_paths.test_data_txt, transform='for_test')
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.PARA.test.batch_size, drop_last=True, shuffle=False, num_workers= cfg.PARA.train.num_workers)
    return test_loader

def Confusion_mxtrix(labels,predicted,num_classes):
    """
    混淆矩阵的函数定义
    Args:
        labels: [num_labels,1] ————10000张图片
        predicted: [num_labels,1] —————10000张图片

    Returns: Confusion_matrix
    """
    Cmatrixs = torch.zeros((num_classes,num_classes))
    stacked = torch.stack((labels, predicted), dim=1)
    for s in stacked:
        a, b = s.tolist()
        Cmatrixs[a, b] = Cmatrixs[a, b] + 1
    return Cmatrixs


def Evaluate(Cmatrixs):
    """for Precision & Recall"""
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_classes = Cmatrixs.size(0)
    Prec, Rec = torch.zeros(n_classes+1), torch.zeros(n_classes+1)

    sum_cmt_row = torch.sum(Cmatrixs,dim=1)#行的和
    sum_cmt_col = torch.sum(Cmatrixs,dim=0)#列的和
    print("----------------------------------------")
    for i in range(n_classes):
        TP = Cmatrixs[i,i]
        FN = sum_cmt_row[i] - TP
        FP = sum_cmt_col[i] - TP
        # TN = torch.sum(Cmatrixs) - sum_cmt_row[i] - FP
        Prec[i] = TP / (TP + FP)
        Rec[i]  = TP / (TP + FN)
        print("%s"%(classes[i]).ljust(10," "),"Presion=%.3f%%,     Recall=%.3f%%"%(Prec[i],Rec[i]))

    Prec[-1] = torch.mean(Prec[0:-1])
    Rec[-1] = torch.mean(Rec[0:-1])
    print("ALL".ljust(10," "),"Presion=%.3f%%,     Recall=%.3f%%" % (Prec[i], Rec[i]))
    print("----------------------------------------")
    # return Prec,Rec


def MyROC_i(outputs, labels, n=20):
    '''
    ROC曲线计算 绘制每一类的
    Args:
        outputs: [num_labels,num_classes]
        labels: 标签值
        n: 得到 n 个点之后绘图
    Returns:plot_roc
    '''

    n_total, n_classes = outputs.size()
    labels = labels.reshape(-1,1) # 行向量转为列向量
    T = torch.linspace(0,1,n)
    TPR, FPR = torch.zeros(n, n_classes+1), torch.zeros(n, n_classes+1)

    for i in range(n_classes):
        for j in range(n):
            mask_1 = outputs[:, i] > T[j]
            TP_FP = torch.sum(mask_1)
            mask_2 = (labels[:, -1] == i)
            TP = torch.sum(mask_1 & mask_2)
            FN = n_total / n_classes - TP
            FP = TP_FP - TP
            TN = n_total - n_total / n_classes - FP

            TPR[j,i] = TP / (TP + FN)
            FPR[j,i] = FP / (FP + TN)

    TPR[:,-1] = torch.mean(TPR[:,0:-1],dim=1)
    FPR[:, -1] = torch.mean(FPR[:, 0:-1], dim=1)

    return TPR,FPR

def Plot_ROC_i(TPR,FPR, args, cfg):
    for i in range(10+1):
        if i==10: width=2
        else: width=1
        plt.plot(FPR[:,i],TPR[:,i],linewidth=width,label='classes_%d'%i)
    plt.legend()
    plt.title("ROC")
    plt.grid(True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(cfg.PARA.utils_paths.visual_path + args.net + '_ROC_i.png')

def plot_confusion_matrix(cm,savename,title='Confusion Matrix'):
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def test(net, epoch, test_loader, log, args, cfg):
    with torch.no_grad():
        labels_value, predicted_value, outputs_value = [],[],[]
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.cuda()
            labels_onehot = labels.cuda()
            _, labels = torch.max(labels_onehot, 1)
            outputs = net(images) #outputs:[100,10]

            _, predicted = torch.max(outputs.data, 1)
            # predicted = ToOnehots(predicted,cfg.PARA.train.num_classes)
            total += labels.size(0)
            correct += (predicted == labels).sum()#.item()

            # Ready for matrixs
            if i==0:
                labels_value = labels
                predicted_value = predicted
                outputs_value = F.softmax(outputs.data,dim=1)
            else:
                labels_value = torch.cat((labels_value,labels),0)
                predicted_value = torch.cat((predicted_value,predicted),0)
                outputs_value = torch.cat((outputs_value,F.softmax(outputs.data,dim=1)),0)

        log.logger.info('epoch=%d,acc=%.5f%%' % (epoch, 100 * correct // total))
        f = open("./cache/visual/"+args.net+"_test.txt", "a")
        f.write("epoch=%d,acc=%.5f%%" % (epoch, 100 * correct // total))
        f.write('\n')

        log.logger.info("==> Get Confusion_Matrixs <==")
        Cmatrixs = Confusion_mxtrix(labels_value,predicted_value,cfg.PARA.train.num_classes)
        # print(Cmatrixs)

        log.logger.info("==> Precision & Recall <==")
        Evaluate(Cmatrixs) #get_Precision & Recall

        log.logger.info("==> Plot_ROC <==")
        TPR_i, FPR_i = MyROC_i(outputs_value, labels_value)
        Plot_ROC_i(TPR_i, FPR_i,args,cfg)

    f.close()

def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger('./cache/log/' + args.net + '_testlog.txt', level='info')
    log.logger.info('==> Preparing data <==')
    test_loader = dataLoad(cfg)
    log.logger.info('==> Loading model <==')
    net = get_network(args,cfg).cuda()
    # net = torch.nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    log.logger.info("==> Waiting Test <==")
    for epoch in range(100, 101):
        # log.logger.info("==> Epoch:%d <=="%epoch)
        checkpoint = torch.load('./cache/checkpoint/'+args.net+'/'+ str(epoch) +'ckpt.pth')
        # checkpoint = torch.load('./cache/checkpoint/' + args.net + '/' + str(60) + 'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        test(net, epoch, test_loader, log, args, cfg)

    log.logger.info('*'*25)

if __name__ == '__main__':
    main()







