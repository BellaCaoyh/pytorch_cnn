#-*- coding:utf-8 _*-
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as mp
import torch
from torchvision import datasets
from torchvision import  transforms
from torch.utils.data import Dataset,DataLoader
from mmcv import Config
from PIL import Image
import argparse
import pdb
from log.logger import Logger

def parser():# set config_path
    parser = argparse.ArgumentParser(description='Dataset PreProcessing')
    parser.add_argument('--config','-c',default='../config/config.py',help='config file path')
    args = parser.parse_args()
    return args

'''1.download dataset'''
def Download_Cifar10(root):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),  # 维度转化 由32x32x3  ->3x32x32
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # R,G,B每层的归一化用到的均值和方差     即参数为变换过程，而非最终结果。
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    '''train_dataset'''
    train_cifar10 = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,  # False,
        transform=transform_train
    )

    '''test_dataset'''
    test_cifar10 = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,  # False,
        transform=transform_test
    )
    return train_cifar10,test_cifar10


'''2.seg dataset to pic'''
def Trainset2Pic(cfg):
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    valid_pic_txt = open(cfg.PARA.cifar10_paths.valid_data_txt, 'w')# 设置为‘w'模式，并且放在最开始，则每次进行时，都会清空重写。
    train_pic_txt = open(cfg.PARA.cifar10_paths.train_data_txt, 'w')
    for i in range(1, 6):
        label_batch = ('A', 'B', 'C', 'D', 'E')
        traindata_file = os.path.join(cfg.PARA.cifar10_paths.original_trainset_path, 'data_batch_' + str(i))
        with open(traindata_file, 'rb') as f:
            train_dict = pkl.load(f,encoding='latin1')  # train_dict为字典，包括四个标签值：b'batch_label',b'labels',b'data',b'filenames'
            data_train = np.array(train_dict['data']).reshape(10000,3,32,32)
            label_train = np.array(train_dict['labels'])
            num_val = int(data_train.shape[0])*cfg.PARA.cifar10_paths.validation_rate


        for j in range(10000):
            imgs = data_train[j]
            i0 = Image.fromarray(imgs[0])
            i1 = Image.fromarray(imgs[1])
            i2 = Image.fromarray(imgs[2])
            img = Image.merge("RGB", (i0, i1, i2))

            picname_valid = cfg.PARA.cifar10_paths.after_validset_path + classes[label_train[j]] + label_batch[i - 1] + str("%05d" % j) + '.png'
            picname_train = cfg.PARA.cifar10_paths.after_trainset_path + classes[label_train[j]] + label_batch[i - 1] + str("%05d" % j) + '.png'

            if j<num_val:
                mp.imsave(picname_valid, img)
                valid_pic_txt.write(picname_valid + ' ' + str(label_train[j]) + '\n')

            else:
                mp.imsave(picname_train, img)
                train_pic_txt.write(picname_train + ' ' + str(label_train[j]) + '\n')

    valid_pic_txt.close()
    train_pic_txt.close()

def Testset2Pic(cfg):
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testdata_file = os.path.join(cfg.PARA.cifar10_paths.original_testset_path, 'test_batch')
    test_pic_txt = open(cfg.PARA.cifar10_paths.test_data_txt, 'w')
    with open(testdata_file, 'rb') as f:
        test_dict = pkl.load(f, encoding='latin1')  # train_dict为字典，包括四个标签值：b'batch_label',b'labels',b'data',b'filenames'
        data_test = np.array(test_dict['data']).reshape(10000, 3, 32, 32)
        label_test= np.array(test_dict['labels'])

    test_pic_txt = open(cfg.PARA.cifar10_paths.test_data_txt, 'a')
    for j in range(10000):
        imgs = data_test[j]
        i0 = Image.fromarray(imgs[0])
        i1 = Image.fromarray(imgs[1])
        i2 = Image.fromarray(imgs[2])
        img = Image.merge("RGB", (i0, i1, i2))

        picname_test = cfg.PARA.cifar10_paths.after_testset_path + classes[label_test[j]] + 'F' + str("%05d" % j) + '.png'
        mp.imsave(picname_test, img)
        test_pic_txt.write(picname_test + ' ' + str(label_test[j]) + '\n')
    test_pic_txt.close()

'''3. pic to dataset'''
class Cifar10Dataset(Dataset):
    def __init__(self,txt,transform):
        super(Cifar10Dataset,self).__init__()
        # with open(txt,'r') as f:
        #     imgs = []
        #     for line in f.readlines():
        #         words = line.strip('\n').split()#去掉每一行的换行符
        #         # pdb.set_trace()
        #         imgs.append((words[0],int(words[1])))

        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            # pdb.set_trace()
            words = line.split()  # 用split将该行切片成列表
            # pdb.set_trace()
            imgs.append((words[0], int(words[1])))
        # pdb.set_trace()
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.imgs[index]
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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])(self.img)

    def transform_test(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])(self.img)


def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger('../cache/log/dataset.txt',level='info')

    # log.logger.info('Downloading dataset')
    # Download_Cifar10(root=cfg.PARA.cifar10_paths.root)
    #
    #
    # log.logger.info('Trainset to Pic & save to txt')
    # Trainset2Pic(cfg)
    # log.logger.info('Testset to Pic & save to txt')
    # Testset2Pic(cfg)

    # log.logger.info('get trainset pic to train_data')
    # train_data = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.train_data_txt, transform='for_train')
    # log.logger.info('get validset pic to valid_data')
    # valid_data = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.valid_data_txt, transform='for_valid')
    log.logger.info('get testset pic to test_data')
    test_data  = Cifar10Dataset(txt=cfg.PARA.cifar10_paths.test_data_txt, transform='for_test')
    # print('num_of_trainData:', len(train_data))
    # print('num_of_valData:', len(valid_data))
    print('num_of_testData:', len(test_data))
    # with open('/home/caoyh/DATASET/cifar10/test.txt','w') as f:

    log.logger.info('*'*25)

if __name__ == '__main__':
    main()





























# root = '/home/caoyh/Dataset/cifar10'
# if not os.path.exists(root): os.makedirs(root)
# dataname = 'cifar-10-batches-py'
# classes = ('airplane', 'car', 'bird', 'cat','deer','dog', 'frog', 'horse', 'ship', 'truck')
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# ])
#
# '''train_dataset'''
# train_cifar_10 = datasets.CIFAR10(
#     root=root,
#     train=True,
#     download=False,#True,
#     transform=transform
# )
#
# '''test_dataset'''
# test_cifar_10 = datasets.CIFAR10(
#     root=root,
#     train=False,
#     download=False,#True,
#     transform=transform
# )
#
# '''train&valid'''
# for i in range(1,6):
#     # print("********************\n")
#     label_batch = ('A','B','C','D','E')
#     traindata_file = os.path.join(root, dataname , 'data_batch_' + str(i))
#     with open(traindata_file, 'rb') as f:
#         train_dict = pkl.load(f,encoding='bytes')# train_dict为字典，包括四个标签值：b'batch_label',b'labels',b'data',b'filenames'
#
#     for j in range(10000):
#         img = np.reshape(train_dict[b'data'][i],(3,32,32))
#         img = img.transpose(1,2,0)
#         picname_valid = root + '/' + 'valid/' + classes[train_dict[b'labels'][j]] + label_batch[i-1] + str("%05d"%j) + '.jpg'
#         picname_train = root + '/' + 'train/' + classes[train_dict[b'labels'][j]] + label_batch[i-1] + str("%05d"%j) + '.jpg'
#
#         if not os.path.exists(root + '/' + 'valid/'):
#             os.makedirs(root + '/' + 'valid/')
#         if not os.path.exists(root + '/' + 'train/'):
#             os.makedirs(root + '/' + 'train/')
#
#         train_pic = open('/home/caoyh/SelfUPUP/Net/cache/train.txt', 'a')
#         valid_pic = open('/home/caoyh/SelfUPUP/Net/cache/valid.txt', 'a')
#
#         if j<= 10000*0.05:
#             mp.imsave(picname_valid,img)
#             valid_pic.write(picname_valid+ ' ' + str(train_dict[b'labels'][j]) + '\n')
#         else:
#             mp.imsave(picname_train,img)
#             train_pic.write(picname_train+ ' ' + str(train_dict[b'labels'][j]) + '\n')
#
# valid_pic.close()
# train_pic.close()
#
# '''test'''
# testdata_file = os.path.join(root, dataname , 'test_batch')
# with open(testdata_file, 'rb') as f:
#     test_xtr = pkl.load(f,encoding='bytes')# train_dict为字典，包括四个标签值：b'batch_label',b'labels',b'data',b'filenames'
#
# for j in range(10000):
#     img = np.reshape(test_xtr[b'data'][j],(3,32,32))
#     img = img.transpose(1,2,0)
#     picname_test = root + '/' + 'test/' + classes[test_xtr[b'labels'][j]] + 'F' + str("%05d"%j) + '.jpg'
#
#     if not os.path.exists(root + '/' + 'test/'):
#         os.makedirs(root + '/' + 'test/')
#
#     test_pic = open( '/home/caoyh/SelfUPUP/Net/cache/test.txt', 'a')
#     mp.imsave(picname_test,img)
#     test_pic.write(picname_test+ ' ' + str(test_xtr[b'labels'][j]) + '\n')
# test_pic.close()
#













