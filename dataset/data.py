from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from mmcv import Config
import argparse
import pdb

def parser():
    parser = argparse.ArgumentParser(description='Data PreProcessing')
    parser.add_argument('--config', '-c', default='../config/config.py', help='config file path')
    args = parser.parse_args()
    return args

def default_loader(path):
    return Image.open(path).convert('RGB')
    #不使用.convert(‘RGB’)进行转换读出来的图像是RGBA四通道的，A通道为透明通道

class CiFar10Dataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        super(CiFar10Dataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()  # 用split将该行切片成列表
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        Trans = DataPreProcess(img)
        if self.transform == 'for_train' or 'for_val':
            img = Trans.transform_train()
        elif self.transform == 'for_test':
            img = Trans.transform_test()
        return img, label

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

def test():
    args = parser()
    cfg = Config.fromfile(args.config)
    train_data = CiFar10Dataset(txt=cfg.PARA.data.train_data_txt, transform='for_train')
    val_data = CiFar10Dataset(txt=cfg.PARA.data.val_data_txt, transform='for_val')
    test_data = CiFar10Dataset(txt=cfg.PARA.data.test_data_txt, transform='for_test')
    print('num_of_trainData:', len(train_data))
    print('num_of_valData:', len(val_data))
    print('num_of_testData:', len(test_data))

def main():
    test()

if __name__ == '__main__':
    main()
