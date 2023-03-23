import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

print(torch.__version__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("device =", device)

n_epochs = 3
batch_size_train = 64   #
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print("batch_idx =", batch_idx)
print(example_targets)
print(example_data.shape)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
# plt.show()
print("that's it ")

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()  # super() 函数是用于调用父类(超类)的一个方法。
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         # 卷积层最重要的可学习参数——权重参数和偏置参数去哪了？
#         # 在Tensorflow中都是先定义好weight和bias，再去定义卷积层的呀！
#         # 别担心，在Pytorch的nn模块中，它是不需要你手动定义网络层的权重和偏置的，这也是体现Pytorch使用简便的地方。
#         # 当然，如果有小伙伴适应不了这种不定义权重和偏置的方法，Pytorch还提供了nn.Functional函数式编程的方法，
#         # 其中的F.conv2d()就和Tensorflow一样，要先定义好卷积核的权重和偏置，作为F.conv2d（）的形参之一。
#         # stride = 1,步长默认为1；
#         # padding = 0,填充为0；
#         #   大多数情况下的kernel_size、padding左右两数均相同，且不采用空洞卷积（dilation默认为1）。
#         #   因此只需要记 O = （I - K + 2P）/ S +1这种在深度学习课程里学过的公式就好了。
#         # dilation = 1,决定了是否采用空洞卷积，默认为1（不采用）。
#         #   从卷积核上的一个参数到另一个参数需要走过的距离，那当然默认是1了，毕竟不可能两个不同的参数占同一个地方吧（为0）
#         # groups = 1,决定了是否采用分组卷积。
#         # bias = TRUE,即是否要添加偏置参数作为可学习参数的一个，默认为True。
#         # padding_mode = 'zeros'，即padding的模式，默认采用零填充。
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         # https://blog.csdn.net/qq_42079689/article/details/102642610
#         # https://blog.csdn.net/qq_43391414/article/details/127227029
#         # Pytorch官网给的输入输出为：
#         # Input: (N, C, H, W) or (C, H, W)
#         # Output: (N, C, H, W) or (C, H, W)(same shape as input)
#         # 将整个通道都有0.5的概率被置为0。如果输入的tensor为3维的话，那么第二维是channel维度，会直接把一整行置0；
#         #   如果输入的tensor为4维的话，那么第二维是channel维度，会直接把一整个二维矩阵（代表宽高）置0
#         #   https://cloud.tencent.com/developer/article/1883945
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         # max_pool2d: 二维池化操作;  https://blog.csdn.net/weixin_38481963/article/details/109962715
#         #   2是kernel_size ：表示做最大池化的窗口大小，可以是单个值，也可以是tuple元组
#         #       这里的kernel_size跟卷积核不是一个东西。 kernel_size可以看做是一个滑动窗口，这个窗口的大小由自己指定。
#         #       如果输入是单个值，例如3，那么窗口的大小就是3 × 3
#         #       还可以输入元组，例如(3, 2) ，那么窗口大小就是3 × 2
#         #       最大池化的方法就是取这个窗口覆盖元素中的最大值。
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         # conv2_drop 即 Dropout2d
#         x = x.view(-1, 320)
#         # https://blog.csdn.net/qq_26400705/article/details/109816853
#         # view()的作用相当于numpy中的reshape，重新定义矩阵的形状。
#         #   例1
#         #   普通用法：
#         #     1 import torch
#         #     2 v1 = torch.range(1, 16)
#         #     3 v2 = v1.view(4, 4)
#         #   其中v1为1 * 16大小的张量，包含16个元素。
#         #   v2为4 * 4大小的张量，同样包含16个元素。注意view前后的元素个数要相同，不然会报错。
#         #   例2
#         #   参数使用 - 1
#         #       1 import torch
#         #       2 v1 = torch.range(1, 16)
#         #       3 v2 = v1.view(-1, 4)
#         #   和图例中的用法一样，view中一个参数定为 -1，代表动态调整这个维度上的元素个数，以保证元素的总数不变。因此两个例子的结果是相同的。
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        input = input.to(device)
        output = self.model(input)
        return F.log_softmax(output,dim=1)


network = Net()
print(network.to(device))

print("here we are")
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata-min)/(max-min)
    return outputdata

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()   # 手动将梯度设置为零
        output = network(data)  # 启动前向传递

        loss = F.nll_loss(output, target)  # 计算输出与真值标签之间的<负对数>概率损失
        # loss = torch.nn.CrossEntropyLoss(output, target)
        loss.backward()     # 反向传播计算
        optimizer.step()    # 将其传播回每个网络参数
        if batch_idx % log_interval == 0:
            print("this is a train()")
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # test_loss += torch.nn.CrossEntropyLoss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print("this is a test()")
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# train(1)
print("that's it 0")

test()  # 不加这个，后面画图就会报错：x and y must be the same size
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# example_data = example_data.to(device)
# example_targets = example_targets.to(device)
with torch.no_grad():
    # output = output.to(device)
    output = network(example_data)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
print("that's it 2")

# ----------------------------------------------------------- #

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
# 使用.load_state_dict()，我们现在可以加载网络的内部状态，并在最后一次保存它们时优化它们。
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

# 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# 不然报错：x and y must be the same size
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
for i in range(4, 9):
    test_counter.append(i * len(train_loader.dataset))
    train(i)
    test()
print("that's it 3")

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
print("that's it 4")

