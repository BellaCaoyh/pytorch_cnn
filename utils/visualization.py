import matplotlib.pyplot as plt
import argparse

def plot_acc_loss(txt,args,cfg,N=0):
    '''
    plot visual pic to save
    :param txt: train_data_txt path
    :param N: from N lines in txt to read
    :param args: get net_name
    :return: save pic to visual
    '''
    EPOCH, ACC_train, ACC_valid, LOSS = [],[],[],[]
    with open(txt, encoding='utf-8') as f:
        for line in f.readlines()[N:]:#从第Nhang开始读取数据
            try:
                temp = line.split(',')
                EPOCH.append(int((temp[0].split('=')[1])))
                ACC_train.append(float(temp[1].split('=')[1]))
                LOSS.append(float(temp[2].split('=')[1]))
            except:
                continue

    plt.plot(EPOCH,ACC_train,color='red',label='acc')
    plt.plot(EPOCH,LOSS,color='blue',label='loss')
    plt.legend(loc=('lower right'))
    plt.grid(True)
    plt.ylim(0,1)
    plt.savefig(cfg.PARA.utils_path.visual_path + args.net + '_train.png')

# def parser():
#     parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
#     # parse.add_argument('--local_rank',default=0,type=int,help='node rank for distributedDataParallel')
#     parse.add_argument('--config','-c',default='./config/config.py',help='config file path')
#     # parse.add_argument('--net','-n',type=str,required=True,help='input which model to use')
#     parse.add_argument('--net','-n',default='resnet50')
#     parse.add_argument('--pretrain','-p',action='store_true',help='Location pretrain data')
#     parse.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
#     parse.add_argument('--epoch','-e',default=None,help='resume from epoch')
#     args = parse.parse_args()
#     # print(argparse.local_rank)
#     return args
#
# args = parser()
# plot_acc_loss(txt='../cache/visual/resnet50_train.txt',args=args)
