import cv2
import pdb
import torch
import torch.nn as nn
from torchvision import  transforms
from torch.utils.data import Dataset,DataLoader

class Normalization():
    def __init__(self):
        super(Normalization, self).__init__()

    def Mean_Std(self, file_path):#计算所有图片的均值和方差
        MEAN_B, MEAN_G, MEAN_R = 0, 0, 0
        STD_B, STD_G, STD_R = 0, 0, 0
        count = 0
        with open(file_path, 'r') as f:
            for line in f.readlines():
                count += 1
                words = line.strip('\n').split()
                img = torch.from_numpy(cv2.imread(words[0])).float()
                MEAN_B += torch.mean(img[:, :, 0] / 255)
                MEAN_G += torch.mean(img[:, :, 1] / 255)
                MEAN_R += torch.mean(img[:, :, 2] / 255)
                STD_B += torch.std(img[:, :, 0] / 255)
                STD_G += torch.std(img[:, :, 1] / 255)
                STD_R += torch.std(img[:, :, 2] / 255)
        MEAN_B, MEAN_G, MEAN_R = MEAN_B / count, MEAN_G / count, MEAN_R / count
        STD_B, STD_G, STD_R = STD_B / count, STD_G / count, STD_R / count
        return (MEAN_B, MEAN_G, MEAN_R), (STD_B, STD_G, STD_R)

    def Normal(self,img, mean, std):#将一张图片，由正态分布变成0-1分布
        img = img/255
        img = img.transpose(2, 0, 1) #[H,W,C]-->[C,H,W],与pytorch的ToTensor做相同的变换，便于比较
        img[0, :, :] =  (img[0, :, :] - mean[0]) / std[0]
        img[1, :, :] =  (img[1, :, :] - mean[1]) / std[1]
        img[2, :, :] =  (img[2, :, :] - mean[2]) / std[2]
        return img


def test():
    img = cv2.imread('/home/caoyh/DATASET/cifar10/trainset/airplaneA00029.png')#读取一张图片，cv读取结果为[b,g,r] [H,W,C]
    my_normal = Normalization()
    img1 = my_normal.Normal(img, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normal = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    converToTensor = transforms.ToTensor()
    img2 = converToTensor(img) #[H,W,C]-->[C,H,W]
    img2 = normal(img2)
    print(img1)
    print(img2)

if __name__ == '__main__':
    test()
