import matplotlib.pyplot as plt
import argparse
from mmcv import Config

def plot_acc_loss(train_txt, valid_txt, args,cfg,N=0):
    '''
    plot visual pic to save
    :param txt: train_data_txt path
    :param N: from N lines in txt to read
    :param args: get net_name
    :return: save pic to visual
    '''
    EPOCH, Loss_train, Loss_valid = [],[],[]
    with open(train_txt, encoding='utf-8') as f:
        for line in f.readlines()[N:]:#从第Nhang开始读取数据
            try:
                temp = line.split(',')
                EPOCH.append(int((temp[0].split('=')[1])))
                Loss_train.append(float(temp[1].split('=')[1]))
                # LOSS.append(float(temp[2].split('=')[1]))
            except:
                continue
    with open(valid_txt, encoding='utf-8') as f:
        for line in f.readlines()[N:]:#从第Nhang开始读取数据
            try:
                temp = line.split(',')
                # EPOCH.append(int((temp[0].split('=')[1])))
                Loss_valid.append(float(temp[1].split('=')[1]))
                # LOSS.append(float(temp[2].split('=')[1]))
            except:
                continue

    plt.plot(EPOCH,Loss_train,color='red',label='Train_Loss')
    plt.plot(EPOCH,Loss_valid,color='blue',label='Valid_Loss')
    plt.legend()
    # plt.legend(loc=('lower right'))
    plt.grid(True)
    # plt.ylim(0,1)
    # plt.savefig(cfg.PARA.utils_paths.visual_path + args.net + '.png')
    plt.savefig('../cache/visual/' + args.net + '.png')
