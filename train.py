import itertools
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch
import datetime
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

CIFAR_PATH = "D:\\Users\\asus\\Desktop\\_ai-cifar100"

batch_size = 200
learning_rate = 0.0001
epochs = 300


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

# trainset = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True)
# train_size = int(0.9 * len(trainset))
# val_size = len(trainset) - train_size
# cifar100_training, cifar100_validating = torch.utils.data.random_split(trainset, [train_size, val_size])
# cifar100_training.dataset.transform = train_transforms
# cifar100_validating.dataset.transform = val_test_transforms


cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH,  train=True, transform=train_transforms)

cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, transform=test_transforms)

cifar100_training.targets = change(cifar100_training.targets)

cifar100_testing.targets = change(cifar100_testing.targets)

train_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=batch_size, shuffle=True)

# val_loader = torch.utils.data.DataLoader(cifar100_validating, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(cifar100_testing, batch_size=batch_size, shuffle=True)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("no")


# 残差块单元residual block
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            # 若input_channels != num_channels则为必须
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # 批量归一化，缓解梯度消失和梯度爆炸，加快收敛
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.dropout(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# 构造stage；num_residuals:一个stage里面包含几个小的残差块（残差块单元级联数量）
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            # stage中的第一级残差块，用stride=2使高宽减半
        else:
            block.append(Residual(num_channels, num_channels))
    return block


# # 搭建Resnet网络
# block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
#                        nn.ReLU())  # nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# block2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
# block3 = nn.Sequential(*resnet_block(64, 128, 4))
# block4 = nn.Sequential(*resnet_block(128, 256, 6))
# block5 = nn.Sequential(*resnet_block(256, 512, 3))
# resnet = nn.Sequential(block1, block2, block3, block4, block5, nn.AdaptiveAvgPool2d((1, 1)),
#                        nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, 20))


# 搭建双流网络?
class DualStreamNet(nn.Module):
    def __init__(self):
        super(DualStreamNet, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
                               nn.ReLU())  # nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
        self.block3 = nn.Sequential(*resnet_block(64, 128, 4))
        self.block4 = nn.Sequential(*resnet_block(128, 256, 6))
        self.block5 = nn.Sequential(*resnet_block(256, 512, 3))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 20)
        self.fc2 = nn.Linear(32*32*64, 20)
        self.classifier = nn.Linear(20 + 20, 20)

    def forward(self, x):
        op1 = self.block1(x)
        op2 = self.block2(op1)
        op1 = self.block3(op2)
        op1 = self.block4(op1)
        op1 = self.block5(op1)
        op1 = self.relu(self.fc1(self.dropout(self.flatten(self.avg_pool(op1)))))
        op2 = self.relu(self.fc2(self.dropout(self.flatten(op2))))
        combined_features = torch.cat((op1, op2), dim=1)
        out = self.classifier(combined_features)
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
            predict = resnet(features)
            Indices = torch.argmax(predict, dim=1)
            ac_total += torch.eq(Indices, labels).sum().item()
            total += len(labels)
            lss = lossF(predict, labels.long())
    return ac_total / total, lss


if __name__ == '__main__':
    # 正常训练
    resnet = DualStreamNet()
    resnet = resnet.to(device)
    # total_params = sum(p.numel() for p in resnet.parameters())
    # print(total_params) # 22605532
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=0.001)

    # # 微调开
    # resnet.load_state_dict(torch.load("./daiding.pth"))
    # for param in resnet.parameters():  # 冻结
    #     param.requires_grad = False
    # for param in resnet[-1].parameters():  # 解冻
    #     param.requires_grad = True

    l1_lambda = 0.0001
    milestones = [200]
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
            predict = resnet(features)
            Indices = torch.argmax(predict, dim=1)
            ls = lossF(predict, labels.long())

            #微调关
            l1_loss = torch.tensor(0., device=device)
            for param in resnet.parameters():
                l1_loss += torch.norm(param, p=1)
            total_loss = ls + l1_lambda * l1_loss
            total_loss.backward()

            # ls.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_acc, train_loss = accuracy(train_loader)
        test_acc, test_loss = accuracy(test_loader)
        train_accuracy_list.append(train_acc)
        train_losses_list.append(train_loss.item())
        test_accuracy_list.append(test_acc)
        test_losses_list.append(test_loss.item())
        et = datetime.datetime.now()
        Time = (et - st).seconds
        scheduler.step()
        print(f"epoch: {epoch + 1}, time:{Time}s, train_loss: {train_loss:.2f}, "
              f"test_loss:{test_loss:.2f}, train_acc: {train_acc :.2%}, test_acc: {test_acc :.2%}")
    torch.save(resnet.state_dict(), "./test.pth")


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
    plt.title('train & test curves')
    plt.show()


# # 单个模型评估准确率
# resnet = DualStreamNet()
# resnet.eval()
# resnet = resnet.to(device)
# resnet.load_state_dict(torch.load("./71.09-74.86(cnn, coarse).pth"))
# total = 0
# all_Indices = []
# all_labels = []
# for i, (features, labels) in enumerate(test_loader):
#     with torch.no_grad():
#         features = features.to(device)
#         labels = labels.to(device)
#         predict = resnet(features)
#         Indices = torch.argmax(predict, dim=1)
#         a = torch.eq(Indices, labels).sum().item() / len(labels)
#         total += a
#         print(f"正确率： {(total / (i + 1)):.2%}")
#         print("预测：", Indices.tolist())
#         print("答案：", labels.tolist())
#         all_Indices.extend(Indices.tolist())
#         all_labels.extend(labels.tolist())


# # 混淆矩阵测试
# target = np.array(all_labels)
# pred = np.array(all_Indices)
# conf_matrix = confusion_matrix(target, pred)
# class_names = ["C" + str(i) for i in range(100)]
# def plot_confusion_matrix(conf_matrix, class_names, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
#     if normalize:
#         conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
#     plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=90)
#     plt.yticks(tick_marks, class_names)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = conf_matrix.max() / 2.
#     for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
#         plt.text(j, i, format(conf_matrix[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if conf_matrix[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# plt.figure(figsize=(10, 10))
# plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix (Unnormalized)')
# plt.show()
