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


os.environ['CUDA_VISION_DEVICES'] = '1'

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

def train(epoch, net, train_loader, args, log, cfg):
    '''Loss & Optimizer'''
    criterion = nn.CrossEntropyLoss().cuda(1)
    optimizer = optim.SGD(net.parameters(), lr=cfg.PARA.train.lr, momentum=cfg.PARA.train.momentum)
    net.train()
    '''Train_Net'''
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    '''Train_model_for one epoch'''
    for i, data in enumerate(train_loader, 0):
        length = len(train_loader) #length = batch_size
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda(1)), Variable(labels.cuda(1))
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # train_loss = train_loss / len(train_loader.dataset)
        if (i+1+epoch*length)%100==0:
            log.logger.info('[Epoch:%d, iter:%d] Loss: %.5f | Acc: %.5f %%'
                        %(epoch+1, (i+1+epoch*length), train_loss/(i+1), 100.*correct/total))

    '''save final acc for visual'''
    with open(cfg.PARA.utils_paths.visual_path+args.net+'_train.txt','a') as f:
        f.write('epoch=%d,acc=%.5f,loss=%.5f\n' %(epoch+1, correct/total, train_loss/length))

    '''save model's net & epoch to checkpoint'''
    log.logger.info('Save model to checkpoint ' )
    checkpoint = {
        'net': net.state_dict(),
        'epoch':epoch
    }
    if not os.path.exists(cfg.PARA.utils_paths.checkpoint_path+args.net):os.makedirs(cfg.PARA.utils_paths.checkpoint_path+args.net)
    torch.save(checkpoint, cfg.PARA.utils_paths.checkpoint_path+args.net+'/'+str(epoch+1)+'ckpt.pth')
    # log.logger.info('Finish save model to checkpoint ')

def valid(epoch, net, valid_loader,log,args):
    # log.logger.info('Begin Validation')
    with torch.no_grad(): #强制之后的内容不进行计算图的构建，不使用梯度反传
        correct = 0.0
        total = 0.0
        net.eval()
        for i,data in enumerate(valid_loader,0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda(1)), Variable(labels.cuda(1))
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        log.logger.info('Validation Acc: %.5f'%(100*correct/total))
    # log.logger.info('Finish Validation')

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

    net = get_network(args).cuda(1)
    # net = resnet18().cuda(1)
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
    for epoch in range(start_epoch, cfg.PARA.train.epoch):
        train(epoch=epoch,net=net,train_loader=train_loader,args=args,log=log,cfg=cfg)
        valid(epoch=epoch,net=net,valid_loader=valid_loader,args=args,log=log)
    log.logger.info('==> Finish Train <==')

    log.logger.info('==> Plot ACC & Save to Visual <==')
    plot_acc_loss(cfg.PARA.utils_paths.visual_path+args.net+'_train.txt', args)
    log.logger.info('*'*25)

if __name__ == '__main__':
    main()







