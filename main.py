import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from models.resnet import resnet50

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='./model/Resnet50.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 135  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.1  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),  # 维度转化 由32x32x3  ->3x32x32
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # R,G,B每层的归一化用到的均值和方差     即参数为变换过程，而非最终结果。
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = resnet50().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题,此标准将LogSoftMax和NLLLoss集成到一个类中。
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
if __name__ == "__main__":
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):  # 从先前次数开始训练
                print('\nEpoch: %d' % (epoch + 1))  # 输出当前次数
                net.train()  # 这两个函数只要适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数
                # 运用net.train()时，训练时每个min - batch时都会根据情况进行上述两个参数的相应调整，所有BatchNormalization的训练和测试时的操作不同。
                sum_loss = 0.0  # 损失数量
                correct = 0.0  # 准确数量
                total = 0.0  # 总共数量
                for i, data in enumerate(trainloader,
                                         0):  # 训练集合enumerate(sequence, [start=0])用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                    # 准备数据  i是序号 data是遍历的数据元素
                    length = len(trainloader)  # 训练数量
                    inputs, labels = data  # data的结构是：[4x3x32x32的张量,长度4的张量]
                    # 假想： inputs是当前输入的图像，label是当前图像的标签，这个data中每一个sample对应一个label
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()  # 清空所有被优化过的Variable的梯度.

                    # forward + backward
                    outputs = net(inputs)  # 得到训练后的一个输出
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()  # 进行单次优化 (参数更新).

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)  # 返回输入张量所有元素的最大值。 将dim维设定为1，其它与输入形状保持一致。
                    # 这里采用torch.max。torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；第二个参数1是代表dim的意思，也就是取每一行的最大值，其实就是我们常见的取概率最大的那个index；第三个参数loss也是torch.autograd.Variable格式。
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():  # 没有求导
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
