#-*- coding:utf-8 _*-
import os
import pdb
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mmcv import Config
from models import resnet
from models.resnet import *
from log.logger import Logger
from dataset.dataset import Cifar10Dataset
from utils.get_net import get_network
from utils.visualization import plot_acc_loss
from loss import MyCrossEntropy


os.environ['CUDA_VISION_DEVICES'] = '0'

assert torch.cuda.is_available(), 'Error: CUDA is not find!'

def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    # parse.add_argument('--local_rank',default=0,type=int,help='node rank for distributedDataParallel')
    parse.add_argument('--config','-c',default='./config/config.py',help='config file path')
    parse.add_argument('--net','-n',type=str,required=True,help='input which model to use')
    # parse.add_argument('--net','-n',default='resnet18')
    parse.add_argument('--pretrain','-p',action='store_true',help='Location pretrain data')
    parse.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
    parse.add_argument('--epoch','-e',default=None,help='resume from epoch')
    parse.add_argument('--gpuid','-g',type=int,default=0,help='GPU ID')
    # parse.add_argument('--NumClasses','-nc',type=int,default=)
    args = parse.parse_args()
    # print(argparse.local_rank)
    return args

def get_model_params(net,args,cfg):
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    with open(cfg.PARA.utils_paths.params_path+args.net+'_params.txt','a') as f:
        f.write('total_params:%d\n'%total_params)
        f.write('total_trainable_params: %d\n'%total_trainable_params)

def DataLoad(cfg):
    trainset = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.train_data_txt, transform='for_train')
    validset = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.valid_data_txt, transform='for_valid')
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.PARA.train.batch_size, drop_last=True, shuffle=True, num_workers=cfg.PARA.train.num_workers)
    valid_loader = DataLoader(dataset=validset, batch_size=cfg.PARA.train.batch_size, drop_last=True, shuffle=True, num_workers=cfg.PARA.train.num_workers)
    return train_loader, valid_loader

def train(net,criterion,optimizer, train_loader, valid_loader, args, log, cfg):
    for epoch in range(cfg.PARA.train.epochs):
        net.train()
        train_loss = 0.0
        train_total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader) #length = 47500 / batch_size
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda(args.gpuid)), Variable(labels.cuda(args.gpuid))
            # pdb.set_trace()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            # pdb.set_trace()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            if (i+1+epoch*length)%100==0:
                log.logger.info('[Epoch:%d, iter:%d] Loss: %.5f '
                            %(epoch+1, (i+1+epoch*length), train_loss/ (i+1)))
        with open(cfg.PARA.utils_paths.visual_path + args.net + '_train.txt', 'a') as f:
            f.write('epoch=%d,loss=%.5f\n' % (epoch + 1, train_loss / length))


        net.eval()
        valid_loss = 0.0
        valid_total = 0.0
        with torch.no_grad():  # 强制之后的内容不进行计算图的构建，不使用梯度反传
            for i, data in enumerate(valid_loader, 0):
                length = len(valid_loader)
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda(args.gpuid)), Variable(labels.cuda(args.gpuid))
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                # correct += (predicted == labels).sum()
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
            log.logger.info('Validation | Loss: %.5f' % (valid_loss / length))
            with open(cfg.PARA.utils_paths.visual_path+args.net+'_valid.txt','a') as f:
                f.write('epoch=%d,loss=%.5f\n' %(epoch+1, valid_loss/length))

        '''save model's net & epoch to checkpoint'''
        log.logger.info('Save model to checkpoint ' )
        checkpoint = { 'net': net.state_dict(),'epoch':epoch}
        if not os.path.exists(cfg.PARA.utils_paths.checkpoint_path+args.net):os.makedirs(cfg.PARA.utils_paths.checkpoint_path+args.net)
        torch.save(checkpoint, cfg.PARA.utils_paths.checkpoint_path+args.net+'/'+str(epoch+1)+'ckpt.pth')


def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path+ args.net + '_trainlog.txt',level='info')
    start_epoch = 0

    log.logger.info('==> Preparing dataset <==')
    train_loader, valid_loader = DataLoad(cfg)

    log.logger.info('==> Loading model <==')
    if args.pretrain:
        log.logger.info('Loading Pretrain Data')

    net = get_network(args, cfg).cuda(args.gpuid)
    criterion = MyCrossEntropy().cuda(args.gpuid)
    optimizer = optim.SGD(net.parameters(), lr=cfg.PARA.train.lr, momentum=cfg.PARA.train.momentum)

    # net = resnet18().cuda(args.gpuid)
    log.logger.info('==> SUM NET Params <==')
    get_model_params(net,args,cfg)

    # if torch.cuda.device_count()>1:#DataParallel is based on Parameter server
    #     net = nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    torch.backends.cudnn.benchmark = True

    '''断点续训否'''
    if args.resume:
        log.logger.info('Resuming from checkpoint')
        checkpoint = torch.load(cfg.PARA.utils_paths.checkpoint_path+args.net+'/'+args.epoch + 'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    log.logger.info('==> Waiting Train <==')
    train(net=net,criterion=criterion,optimizer=optimizer,
          train_loader=train_loader,valid_loader=valid_loader,args=args,log=log,cfg=cfg)
    log.logger.info('==> Finish Train <==')

    log.logger.info('==> Plot Train_Vilid Loss & Save to Visual <==')
    plot_acc_loss(args, cfg=cfg)
    log.logger.info('*'*25)

if __name__ == '__main__':
    main()







