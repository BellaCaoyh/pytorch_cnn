import os
import pickle as p
import numpy as np
import pdb
import argparse
from mmcv import Config
from PIL import Image

INDEX = ['A','B','C','D','E']
aa=0

def parser_args():
    parser = argparse.ArgumentParser(description='Segment Dataset')
    parser.add_argument('--config', '-c', default='../config/config.py', help='config file path')
    args = parser.parse_args()
    return args

def SegDataset_ToPic(path, filename,cfg):
    global aa
    file = os.path.join(path, filename) # 路径拼接
    with open(file, 'rb')as f:
        datadict = p.load(f, encoding='latin1')
        x_train = datadict['data']
        t_train = datadict['labels']
        x_train = np.array(x_train)
        t_train = np.array(t_train)
        validation_num = int(x_train.shape[0] * cfg.PARA.data.rate)
        x_train=x_train.reshape(10000, 3, 32, 32)

    label_dict = {0 : 'airplane', 1 : 'automobile', 2 : 'bird',  3 : 'cat', 4 : 'deer', 5 : 'dog', 6 : 'frog', 7 : 'horse', 8 : 'ship', 9 : 'truck' }

    for i in range(validation_num):
        imgs = x_train[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = str(label_dict[t_train[i]]) + str(i) + str(INDEX[aa]) + '.png'
        img.save(cfg.PARA.data.after_valset_path + name, "png")

    for i in range(validation_num,10000):
        imgs = x_train[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = str(label_dict[t_train[i]]) + str(i)+ str(INDEX[aa])  + '.png'
        img.save(cfg.PARA.data.after_trainset_path + name, "png")
    aa = aa + 1


def Testset_ToPic(path, cfg):
    with open(path, 'rb')as f:
        datadict = p.load(f, encoding='latin1')
        x_test = datadict['data']
        t_test = datadict['labels']
        t_test = np.array(t_test)
        x_test = x_test.reshape(10000, 3, 32, 32)

    label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    for i in range(10000):
        imgs = x_test[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = str(label_dict[t_test[i]]) + str(i)  + '.png'
        img.save(cfg.PARA.data.after_testset_path + name, "png")

def main():
    args = parser_args()
    cfg = Config.fromfile(args.config)
    train_batch_path = cfg.PARA.data.original_trainset_path
    child_path = os.listdir(train_batch_path)
    child_path.sort()
    for filename in child_path:
        SegDataset_ToPic(train_batch_path, filename, cfg)
    Testset_ToPic(cfg.PARA.data.original_testset_path,cfg)
    #生成.txt文件
    dir = {0: cfg.PARA.data.after_trainset_path, 1: cfg.PARA.data.after_valset_path, 2: cfg.PARA.data.after_testset_path}
    NAME = {0: 'train', 1: 'val', 2: 'test'}
    TXT = {0: cfg.PARA.data.train_data_txt, 1: cfg.PARA.data.val_data_txt, 2: cfg.PARA.data.test_data_txt}
    for i in range(3):
        files = os.listdir(dir[i])
        NAME[i] = open(TXT[i], 'a')
        for file in files:
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            temp1 = fileType[1].split('.')[0]
            temp2 = temp1.rstrip('0123456789ABCDE')
            label_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                          'horse': 7, 'ship': 8, 'truck': 9}
            label = label_dict[temp2]
            name = str(dir[i]) + file + ' ' + str(int(label)) + '\n'
            NAME[i].write(name)
        NAME[i].close()

if __name__ == "__main__":
    main()