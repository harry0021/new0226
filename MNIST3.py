import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot

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
        output = self.model(input)
        return output


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device =",device)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

BATCH_SIZE = 256
EPOCHS = 10
trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)

trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)
net = Net()
print(net.to(device))

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

history = {'Test Loss': [], 'Test Accuracy': []}

#开始对训练集的DataLoader进行迭代
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):

        # 将图像和标签传输进device中
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        # 清空模型的梯度
        net.zero_grad()

        # 对模型进行前向推理
        outputs = net(trainImgs)

        # 计算本轮推理的Loss值
        loss = lossF(outputs, labels)

        # 计算本轮推理的准确率
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            net.train(False)
            with torch.no_grad():
                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)

                    totalLoss += loss
                    correct += torch.sum(predictions == labels)

                    testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                    testLoss = totalLoss / len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())

            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                    (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))
    processBar.close()

matplotlib.pyplot.plot(history['Test Loss'], label='Test Loss')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Loss')
matplotlib.pyplot.show()

matplotlib.pyplot.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Accuracy')
matplotlib.pyplot.show()

torch.save(net, './model.pth')

