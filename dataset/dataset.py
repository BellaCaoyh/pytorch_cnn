#-*- coding:utf-8 _*-
import os
import pickle as pkl
import numpy as np
import random
import matplotlib.pyplot as mp
import torch
from torchvision import datasets
from torchvision import  transforms
from torch.utils.data import Dataset,DataLoader
from mmcv import Config
from PIL import Image
import argparse
import pdb
import cv2
from log.logger import Logger
from utils import ToOnehot

def parser():# set config_path
    parser = argparse.ArgumentParser(description='Dataset PreProcessing')
    parser.add_argument('--config','-c',default='../config/config.py',help='config file path')
    args = parser.parse_args()
    return args

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

'''2.seg dataset to pic'''
def Trainset2Pic(cfg):
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    valid_pic_txt = open(cfg.PARA.cifar10_paths.valid_data_txt, 'w')# 设置为‘w'模式，并且放在最开始，则每次进行时，都会清空重写。
    train_pic_txt = open(cfg.PARA.cifar10_paths.train_data_txt, 'w')
    for i in range(1, 6):
        label_batch = ('A', 'B', 'C', 'D', 'E')
        traindata_file = os.path.join(cfg.PARA.cifar10_paths.original_trainset_path, 'data_batch_' + str(i))
        with open(traindata_file, 'rb') as f:
            train_dict = pkl.load(f,encoding='bytes')  # encoding=bytes ,latin1,train_dict为字典，包括四个标签值：b'batch_label',b'labels',b'data',b'filenames'
            data_train = np.array(train_dict[b'data']).reshape(10000, 3, 32, 32)
            label_train = np.array(train_dict[b'labels'])
            num_val = int(data_train.shape[0]*cfg.PARA.cifar10_paths.validation_rate)#验证集的个数

            val_list = random.sample(list(range(0,int(data_train.shape[0]))), num_val)

        for j in range(10000):
            imgs = data_train[j]
            r = Image.fromarray(imgs[0])
            g = Image.fromarray(imgs[1])
            b = Image.fromarray(imgs[2])
            img = Image.merge("RGB", (b, g, r))

            picname_valid = cfg.PARA.cifar10_paths.after_validset_path + classes[label_train[j]] + label_batch[i - 1] + str("%05d" % j) + '.png'
            picname_train = cfg.PARA.cifar10_paths.after_trainset_path + classes[label_train[j]] + label_batch[i - 1] + str("%05d" % j) + '.png'

            if (j in val_list):#如果在随机选取的验证集列表中，则保存到验证集的文件夹下
                img.save(picname_valid)
                valid_pic_txt.write(picname_valid + ' ' + str(label_train[j]) + '\n')

            else:
                img.save(picname_train)
                train_pic_txt.write(picname_train + ' ' + str(label_train[j]) + '\n')

    valid_pic_txt.close()
    train_pic_txt.close()

def Testset2Pic(cfg):
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testdata_file = os.path.join(cfg.PARA.cifar10_paths.original_testset_path, 'test_batch')
    test_pic_txt = open(cfg.PARA.cifar10_paths.test_data_txt, 'w')
    with open(testdata_file, 'rb') as f:
        test_dict = pkl.load(f, encoding='bytes')  # train_dict为字典，包括四个标签值：b'batch_label',b'labels',b'data',b'filenames'
        data_test = np.array(test_dict[b'data']).reshape(10000, 3, 32, 32)
        label_test= np.array(test_dict[b'labels'])

    test_pic_txt = open(cfg.PARA.cifar10_paths.test_data_txt, 'a')
    for j in range(10000):
        imgs = data_test[j]

        r = Image.fromarray(imgs[0])
        g = Image.fromarray(imgs[1])
        b = Image.fromarray(imgs[2])
        img = Image.merge("RGB", [b, g, r])

        picname_test = cfg.PARA.cifar10_paths.after_testset_path + classes[label_test[j]] + 'F' + str("%05d" % j) + '.png'
        img.save(picname_test)
        test_pic_txt.write(picname_test + ' ' + str(label_test[j]) + '\n')
    test_pic_txt.close()

'''3. pic to dataset'''
class Cifar10Dataset(Dataset):
    def __init__(self,txt,transform):
        super(Cifar10Dataset,self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()  # 用split将该行切片成列表
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.imgs[index]
        label = ToOnehot(label,10)
        img = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            Trans = DataPreProcess(img)
            if self.transform == 'for_train' or 'for_valid':
                img = Trans.transform_train()
            elif self.transform == 'for_test':
                img = Trans.transform_test()
        return img,label

    def __len__(self):
        return len(self.imgs)

class DataPreProcess():
    def __init__(self,img):
        self.img = img

    def transform_train(self):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#三个均值，三个方差。
        ])(self.img)

    def transform_test(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])(self.img)


def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    # log = Logger('../cache/log/dataset.txt',level='info')

    # log.logger.info('Downloading dataset')
    # Download_Cifar10(root=cfg.PARA.cifar10_paths.root)
    #
    #
    # log.logger.info('Trainset to Pic & save to txt')
    Trainset2Pic(cfg)
    # log.logger.info('Testset to Pic & save to txt')
    Testset2Pic(cfg)

    # log.logger.info('get trainset pic to train_data')
    # train_data = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.train_data_txt, transform='for_train')
    # log.logger.info('get validset pic to valid_data')
    # valid_data = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.valid_data_txt, transform='for_valid')
    # log.logger.info('get testset pic to test_data')
    # test_data  = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.test_data_txt, transform='for_test')
    # print('num_of_trainData:', len(train_data))
    # print('num_of_valData:', len(valid_data))
    # print('num_of_testData:', len(test_data))
    # with open('/home/caoyh/DATASET/cifar10/test.txt','w') as f:

    # log.logger.info('*'*25)
    # x = np.arange(0,100)
    # print(x)
    # x1 = np.random.shuffle(x)
    # print(x)
    # y = x1[0:10]
    # print(y)
if __name__ == '__main__':
    main()




