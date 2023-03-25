import torch
import torchvision
from torch.utils.data import DataLoader
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
    torchvision.datasets.MNIST('./data/', train=True, download=True,    # dataset是一个个类的实例
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                                   # Normalize()转换使用的值0.1307和0.3081是MNIST数据集的全局平均值和标准偏差，这里我们将它们作为给定值。
                                   # 表明是个tuple，既存图片又存标签
                               ])),
    batch_size=batch_size_train, shuffle=True)   # shuffle=True用于打乱数据集，每次都会以不同的顺序返回

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
# dataloader是一个类（可迭代iterable，用for循环，用next()遍历）

print("batch_idx =", batch_idx)     # batch_idx = 0
print(example_targets)  # tensor([3,9,...,2
#                                 x,x,...,x
#                                 x,x,...,2])共
print(example_data.shape)   # torch.Size([1000, 1, 28, 28])

#####################
for batch in train_loader:
    print("type(batch): {}".format(type(batch)))  # <class 'list'>
    print("len(batch): {}".format(len(batch)))  # 2
    print("type(batch[0]): {}".format(type(batch[0])))  # <class 'torch.Tensor'>
    print("type(batch[1]): {}".format(type(batch[1])))  # <class 'torch.Tensor'>
    print("batch[0].shape: {}".format(batch[0].shape))  # torch.Size([64, 1, 28, 28])
    print("batch[1].shape: {}".format(batch[1].shape))  # torch.Size([64])  # 64个图像的标签
    break

print("len(train_loader): {}".format(len(train_loader)))  # 938 = [60000/64]
print("len(train_loader.dataset): {}".format(len(train_loader.dataset)))  # 60000

for batch, (x, y) in enumerate(train_loader):
    print("batch: {}, type(x): {}, type(y): {}".format(batch, type(x), type(y)))
    # batch: 0, type(x): <class 'torch.Tensor'>, type(y): <class 'torch.Tensor'>
    # batch是下标索引，(x,y)这个tuple就是数据本身
    print("batch: {}, x.shape: {}, y.shape: {}".format(batch, x.shape, y.shape))
    # batch: 0, x.shape: torch.Size([10000, 1, 28, 28]), y.shape: torch.Size([10000])
    break
####################

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
print("that's it ")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # padding = 1 维持了输出尺寸不变
            # https://blog.csdn.net/weixin_42899627/article/details/108228008
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(),

            torch.nn.Flatten(),  # 打平成7*7*64 = 3136的向量

            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=128, out_features=10),
            # (7*7*64 * 128) + (128 * 10) = 402688, 为参数量
            # 然后每个参数是用一个4字节的浮点数来表示，所以约为1.6MB
            # 因此需要约1.6M的显存

            torch.nn.Softmax(dim=1)
            # Softmax =多类别分类问题=只有一个正确答案=互斥输出
            # Softmax函数是二分类函数Sigmoid在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。
            # Softmax直白来说就是将原来输出通过Softmax函数一作用，就映射成为(0,1)的值，
            # 而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率，
            # 在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标。
        )

    def forward(self, input):
        input = input.to(device)
        output = self.model(input)
        return F.log_softmax(output, dim=1)
    # https://blog.csdn.net/weixin_42214565/article/details/102381380
    # Pytorch训练网络模型过程中Loss为负值的问题及其解决方案


network = Net()
print(network.to(device))

print("here we are")
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# SGD是最基础的优化方法，普通的训练方法, 需要重复不断的把整套数据放入神经网络NN中训练, 这样消耗的计算资源会很大.
#   当我们使用SGD会把数据拆分后再分批不断放入 NN 中计算.
#   每次使用批数据, 虽然不能反映整体数据的情况, 不过却很大程度上加速了 NN 的训练过程, 而且也不会丢失太多准确率.
# Momentum :
#   传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 此方法比较曲折。
#   在普通的梯度下降法x+=v中，每次x的更新量v为 v=−dx∗lr，其中dx为目标函数func(x)对x的一阶导数，
#   当使用冲量时，则把每次x的更新量v考虑为本次的梯度下降量−dx∗lr与上次x的更新量v乘上一个介于[0,1]因子momentum的和
#   即 v=−dx∗lr+v∗ momentum
# learning rate:
#    学习率较小时，收敛到极值的速度较慢。
#    学习率较大时，容易在搜索过程中发生震荡。
# https://blog.csdn.net/apsvvfb/article/details/72536495
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    # model.train()的作用是启用 Batch Normalization 和 Dropout。
    # model.train()是保证BN层能够用到每一批数据的均值和方差。
    # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()   # 手动将梯度设置为零

        output = network(data)  # 启动前向传递

        loss = F.nll_loss(output, target)  # 计算输出与真值标签之间的<负对数>概率损失

        loss.backward()     # 反向传播计算

        optimizer.step()    # 将其传播回每个网络参数

        if batch_idx % log_interval == 0:

            print("output.shape: {}".format(output.shape))
            print("output = ", output)
            print("loss.shape: {}".format(loss.shape))
            print("loss = ", loss)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
            # 神经网络模块以及优化器能够使用.state_dict()保存和加载它们的内部状态。
            # 这样，如果需要，我们就可以继续从以前保存的状态dict中进行训练——只需调用.load_state_dict(state_dict)。

def test():
    network.eval()
    # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
    # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
    # 对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # model.eval()和torch.no_grad()的区别:
        # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); BN层会继续计算数据的mean和var等参数并更新。
        # 在eval模式下，dropout层会让所有的激活单元都通过，而BN层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
        #   如果不在意显存大小和计算时间的话，仅仅使用model.eval()已足够得到正确的validation/test的结果；
        #   而with torch.no_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储梯度），从而可以更快计算，也可以跑更大的batch来测试。
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    # print("this is a test()")
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# train(1)
print("that's it 0")

test()  # 不加这个，后面画图就会报错：x and y must be the same size

print("this is an initial test")

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
