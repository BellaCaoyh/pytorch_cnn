import torch
import pdb
import argparse
from models import *
from dataset.dataset import Cifar10Dataset
from mmcv import Config
from torch.utils.data import DataLoader
from log.logger import Logger
from utils.get_net import get_network
from utils.utils import ToOnehots
import torchvision
import torchvision.transforms as transforms

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

def test(net, epoch, test_loader, log, args, cfg):
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            _, labels = torch.max(labels, 1)
            # pdb.set_trace()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = ToOnehots(predicted,cfg.PARA.train.num_classes)
            total += labels.size(0)
            correct += (predicted == labels).sum()#.item()
        log.logger.info('epoch=%d,acc=%.5f%%' % (epoch, 100 * correct // total))
        f = open("./cache/visual/"+args.net+"_test.txt", "a")
        f.write("epoch=%d,acc=%.5f%%" % (epoch, 100 * correct // total))
        f.write('\n')
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
    for epoch in range(30, 101):
        # log.logger.info("==> Epoch:%d <=="%epoch)
        checkpoint = torch.load('./cache/checkpoint/'+args.net+'/'+ str(epoch) +'ckpt.pth')
        # checkpoint = torch.load('./cache/checkpoint/' + args.net + '/' + str(60) + 'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        test(net, epoch, test_loader, log, args, cfg)

if __name__ == '__main__':
    main()







