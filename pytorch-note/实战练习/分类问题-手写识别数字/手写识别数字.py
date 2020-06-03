import torch
import torchvision
import numpy as np
from utils import *
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.nn import functional as f

learning_rate = 0.01
batch_size = 512

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_data = torchvision.datasets.MNIST('./data', train=True, transform=transforms, download=True)
test_data = torchvision.datasets.MNIST('./data', train=False, transform=transforms, download=True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# x, y = next(iter(train_dataloader))
# plot_image(x,y,"train")
# print(x.shape, y.shape)

# torch.Size([512, 1, 28, 28]) torch.Size([512])  size中第一个维度是图片的数量，第二个维度是通道数，后面两个维度是通道大小

class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()

        # x@w1 + b1
        self.fc1 = nn.Linear(28 * 28, 256)  # 256由经验决定
        self.fc2 = nn.Linear(256, 64)  # 64也是经验决定
        self.fc3 = nn.Linear(64, 10)  # 第二个参数是识别类别的数量

    def forward(self, x):
        # x: [b, 1, 28, 28] 有b张图片
        # h1 = relu(xw1+b1)
        x = f.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = f.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x


linearnet = LinearNet()


# train code
# 逻辑：每一次求导，然后再更新数值
def train():
    # [w1,b1,w2,b2,w3,b3]
    optimizer = optim.SGD(linearnet.parameters(), lr=learning_rate, momentum=0.9)
    train_loss = []
    for epoch in range(1, 4):
        for batch, (x, y) in enumerate(train_dataloader):
            # x: img y: label
            # x: [img_num, 1, 28, 28], y:[img_num]

            # 网络只接受[img_num, 特征通道数]大小的
            # [img_num, 1, 28, 28]=>[img_num, 特征通道数]
            x = Variable(x)
            y = Variable(y)
            x = x.view(x.size(0), 28 * 28)
            # =>[img_num, 10]
            out = linearnet(x)
            y_onehot = one_hot(y, depth=10)
            # loss = mse(out, y_onehot)
            loss = f.mse_loss(out, y_onehot)

            # 清零梯度
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # w' = w - lr*grad
            optimizer.step()
            train_loss.append(loss.item())

            if batch % 10 == 0:
                print("Epoch: %d\tBatch: %d\tLoss: %f" % (epoch, batch, loss.item()))
    # 得到w1,b1,w2,b2,w3,b3
    plot_curve(train_loss)
    torch.save(linearnet, 'model_.pkl')



# 准确度测试
total_correct = 0
model = torch.load('model_.pkl')
for x, y in test_dataloader:
    x = x.view(x.size(0), 28 * 28)
    out = linearnet(x)
    # out :[img_num, 10]=> pred:[img_num]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_dataloader.dataset)
acc = total_correct / total_num
print("test acc: %f" % acc)

# 输出结果
x, y = next(iter(test_dataloader))
x = x.view(x.size(0), 28 * 28)
out = linearnet(x)
pred = out.argmax(dim=1)
# plot_image(x, pred, "real")