import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
from train import DualStreamNet

CIFAR_PATH = "D:\\Users\\asus\\Desktop\\_ai-cifar100"

batch_size = 200
learning_rate = 0.001
epochs = 200


# 小类标签变为大类标签
def change(targets):
    coarse_labels = np.array([
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
        18, 1, 2, 15, 6, 0, 17, 8, 14, 13
    ])
    return coarse_labels[targets]


train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomRotation(5),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
])

cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH,  train=True, transform=train_transforms)

cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, transform=test_transforms)

cifar100_training.targets = change(cifar100_training.targets)

cifar100_testing.targets = change(cifar100_testing.targets)

train_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(cifar100_testing, batch_size=batch_size, shuffle=True)


# file = "./cifar-100-python/train"
# import pickle
# with open(file, 'rb') as fo:
#     dict = pickle.load(fo, encoding='bytes')
# cifar = dict.keys()
# print(cifar)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("no")


# class cnn_feature(nn.Module):
#     def __init__(self):
#         super(cnn_feature, self).__init__()
#         self.block1 = DualStreamNet().block1
#         self.block2 = DualStreamNet().block2
#
#     def forward(self, x):
#         out = self.block1(x)
#         out = self.block2(out)
#         return out


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(3*32, 256, batch_first=True, num_layers=4, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256*2, 20)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(2*4, x.size(0), 256).to(x.device)
        c0 = torch.zeros(2*4, x.size(0), 256).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 只使用最后一个时间步输出作为分类结果
        out = self.relu(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out


def accuracy(loader):
    ac_total = 0
    total = 0
    for i, (features, labels) in enumerate(loader):
        with torch.no_grad():
            if i > 10:
                break
            features = features.to(device)
            labels = labels.to(device)
            batch_size, channels, height, width = features.shape
            # features = cnn(features)
            features = features.view(batch_size, -1, 3*32)
            predict = model(features)
            Indices = torch.argmax(predict, dim=1)
            ac_total += torch.eq(Indices, labels).sum().item()
            total += len(labels)
            lss = lossF(predict, labels.long())
    return ac_total / total, lss


if __name__ == '__main__':
    # cnn = cnn_feature()
    model = LSTMModel()
    # cnn = cnn.to(device)
    model = model.to(device)
    lossF = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    l1_lambda = 0.00001
    milestones = [125, 175]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    st = datetime.datetime.now()
    total_loss, ls = 0, 0
    train_accuracy_list = []
    train_losses_list = []
    test_accuracy_list = []
    test_losses_list = []
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            batch_size, channels, height, width = features.shape
            # features = cnn(features)
            features = features.view(batch_size, -1, 3*32)
            outputs = model(features)
            ls = lossF(outputs, labels.long())

            # l1_loss = torch.tensor(0., device=device)
            # for param in model.parameters():
            #     l1_loss += torch.norm(param, p=1)
            # total_loss = ls + l1_lambda * l1_loss
            # total_loss.backward()

            ls.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_acc, train_loss = accuracy(train_loader)
        test_acc, test_loss = accuracy(test_loader)
        train_accuracy_list.append(train_acc)
        train_losses_list.append(train_loss.item())
        test_accuracy_list.append(test_acc)
        test_losses_list.append(test_loss.item())
        et = datetime.datetime.now()
        scheduler.step()
        Time = (et - st).seconds
        print(f"epoch: {epoch + 1}, time:{Time}s, train_loss: {train_loss:.2f}, "
              f"test_loss:{test_loss:.2f}, train_acc: {train_acc :.2%}, test_acc: {test_acc :.2%}")
    torch.save(model.state_dict(), "./lstm.pth")

    # 图像绘制
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.plot(range(1, epochs + 1), train_losses_list, label='train loss', color='tab:red')
    ax1.plot(range(1, epochs + 1), test_losses_list, label='test loss', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax2.plot(range(1, epochs + 1), train_accuracy_list, label='train acc', color='tab:orange')
    ax2.plot(range(1, epochs + 1), test_accuracy_list, label='test acc', color='tab:green')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='upper right')
    plt.title('train & test curves (lstm)')
    plt.show()


# # 单个评估准确率
# model = LSTMModel()
# model.eval()
# model = model.to(device)
# model.load_state_dict(torch.load("./49.33-83.5(lstm, coarse).pth"))
# total = 0
# all_Indices = []
# all_labels = []
# for i, (features, labels) in enumerate(test_loader):
#     with torch.no_grad():
#         features = features.view(batch_size, -1, 3*32)
#         features = features.to(device)
#         labels = labels.to(device)
#         predict = model(features)
#         Indices = torch.argmax(predict, dim=1)
#         a = torch.eq(Indices, labels).sum().item() / len(labels)
#         total += a
#         print(f"正确率： {(total / (i + 1)):.2%}")
#         print("预测：", Indices.tolist())
#         print("答案：", labels.tolist())
