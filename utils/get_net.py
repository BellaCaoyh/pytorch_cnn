import os
import sys
from models import resnet,vgg,inception,squeezenet
from models.resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from models.vgg import vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn
from models.inception import inception_v3
from models.squeezenet import squeezenet1_0
from models.alexnet import alexnet
from models.lenet5 import LeNet5
import pdb

def get_network(args,cfg):
    """ return given network
    """
    # pdb.set_trace()
    if args.net == 'lenet5':
        net = LeNet5().cuda()
    elif args.net == 'alexnet':
        net = alexnet(pretrained=args.pretrain, num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg16':
        net = vgg16(pretrained=args.pretrain, num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg13':
        net = vgg13(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg11':
        net = vgg11(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg19':
        net = vgg19(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg16_bn':
        net = vgg16_bn(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg13_bn':
        net = vgg13_bn(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg11_bn':
        net = vgg11_bn(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'vgg19_bn':
        net = vgg19_bn(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net =='inceptionv3':
        net = inception_v3().cuda()
    # elif args.net == 'inceptionv4':
    #     net = inceptionv4().cuda()
    # elif args.net == 'inceptionresnetv2':
    #     net = inception_resnet_v2().cuda()
    elif args.net == 'resnet18':
        net = resnet18(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda(args.gpuid)
    elif args.net == 'resnet34':
        net = resnet34(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'resnet50':
        net = resnet50(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda(args.gpuid)
    elif args.net == 'resnet101':
        net = resnet101(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'resnet152':
        net = resnet152(pretrained=args.pretrain,num_classes=cfg.PARA.train.num_classes).cuda()
    elif args.net == 'squeezenet':
        net = squeezenet1_0().cuda()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
