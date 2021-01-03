#-*- coding:utf-8 _*-
from model import resnet
from dataset.dataset import Cifar10Dataset
from utils import get_net
import os
import pdb
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from mmcv import Config
from log.logger import Logger

def parser():
    parser = argparse.ArgumentParser(description='Pytorch Cifar10 Testing')
    parser.add_argument('--config','-c',default='./config/config.py',help='config file path')
    parser.add_argument('--net','-n',type=str, required= True,help='input which model to test')
    args = parser.parse_args()
    return args

def DataLoad(cfg):
    test_data = Cifar10Dataset(txt=cfg.PARA.data.test_data_txt, transform='for_test')
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.PARA.test.BATCH_SIZE, drop_last=True, shuffle=False,num_workers=cfg.PARA.train.num_workers)
    return test_loader

def test(epoch, net,test_loader,log, args,cfg):
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        log.logger.infor('epoch=%d, acc=%.5f%%'%(epoch, 100*correct/total))
        with open(cfg.PARA.utils_paths.visual_path+args.net+'_test_txt','a') as f:
            f.write('epoch=%d,acc=%.5f\n' % (epoch, correct/total))

def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path + args.net + '_testlog.txt',level='info')
    log.logger.info('==> Preparing dataset <==')
    test_loader = DataLoad(cfg=cfg)
    log.logger.info('==> Loading model <==')
    net = get_net(args).cuda()
    if torch.cuda.device_count()>1:#DataParallel is based on Parameter server
        net = nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    log.logger.info('==> Waiting Test <==')
    for epoch in range(1, cfg.PARA.train.epoch+1):
        checkpoint = torch.load(cfg.PARA.utils_paths.checkpoint_path + args.net + '/' + str(epoch) + 'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        test(net,epoch,test_loader,log,args,cfg)
    log.logger.info('==> Finish Test <==')

if __name__ == '__main__':
    main()
    





