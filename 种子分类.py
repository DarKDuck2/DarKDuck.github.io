import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

image_depth = 3
image_size = 224
path = "/kaggle/input/neu-plantseedlingsclassificationdl/Nonsegmented_pack - k"
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# pil - > tensor
dataset = torchvision.datasets.ImageFolder(path + '/train', transforms)

train_ratio = 0.8  # 80% 的数据用于训练，20% 用于验证
dataset_size = len(dataset)

train_size = int(train_ratio * dataset_size)
valid_size = dataset_size - train_size
import random
from torch.utils.data import random_split

random_seed = 25
random.seed(random_seed)
# 划分数据集为训练集和验证集
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

from torch.utils.data import DataLoader

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
print("dataload over")

from torchvision import models
from torch import nn
import torch
import timm

resnet50 = timm.create_model('tf_efficientnet_b8', checkpoint_path='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b8/1/tf_efficientnet_b8_ra-572d5dd9.pth',pretrained = True)
sum = 0
for i,param in enumerate(resnet50.parameters()):
    sum += 1
    if i < 30:
        param.requires_grad = False
print(sum)
num_classes = 12  # 你的任务的类别数量

resnet50.classifier = nn.Sequential(
            nn.Linear(resnet50.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

print(resnet50)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50
model #下面要写训练了
from sklearn.metrics import recall_score, f1_score
num_epochs = 20           #训练轮次
train_loss_list = []       #损失集合
train_acc = 0              #训练准确率
count1 = len(train_loader.dataset)  #dataset的数量
count2 = len(val_loader.dataset)

for epoch in range(num_epochs):
    model.train()
    print(epoch+1)
    poch_acc = 0
    train_loss = 0
    epoch_acc = 0
    for images, labels in tqdm(train_loader):
        #images = images.view(-1,1,28,28)
        #count += len(labels)
        images, labels = images.to(device), labels.to(device)
        #把数据转移到gpu
        images = images.float()
        #变成float
        y_pred = model(images)
        #模型计算结果
        loss = loss_fn(y_pred, labels)
        #计算损失
        optim.zero_grad()
        #计算梯度
        loss.backward()
        #反向传播
        optim.step()

        train_loss += loss.item()
        #计算loss
        epoch_acc += (y_pred.argmax(1) == labels).sum().item()
        #计算准确率
    print('Epoch: {} - Loss: {:.6f}, Training Acc.: {:.3f}'.format(epoch + 1,train_loss / count1,epoch_acc / count1))
    val_acc = 0
    model.eval()
    labels_list=[]
    y_pred_list=[]
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.float()
        y_pred = model(images)
        y_pred_list.extend(y_pred.argmax(1) .tolist())
        labels_list.extend(labels.tolist())
        val_acc += (y_pred.argmax(1) == labels).sum().item()

    y_true = np.array(labels_list)
    y_pred_classes = np.array(y_pred_list)

    recall = recall_score(y_true, y_pred_classes, average='micro')
    f1 = f1_score(y_true, y_pred_classes, average='micro')

    print('Epoch: {} , Val Acc.: {:.3f}'.format(epoch + 1, val_acc / count2))
    print('Recall: {:.6f},F1:{:.6f}'.format(recall, f1))
    model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

#######################################测试集构建############################################
# Test，得把test数据也封装成等batch_size的loader

print('test文件夹文件读取中……')
files = os.listdir(path + '/test')
print(files)
test_dataset = list()
for i in tqdm(files):
    image = Image.open(path + '/test/' + i)
    image = image.resize((image_size, image_size))
    image = np.array(image)
    to_tensor = torchvision.transforms.ToTensor()
    image_tensor = to_tensor(image)
    norm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image_tensor = norm(image_tensor)
    test_dataset.append(image_tensor)
testdata = files
print(testdata)
print(f'数据加载完成，train图片数量为：{len(test_dataset)}')
import torch
from torch.utils.data import Dataset, DataLoader

test_dataset = np.array(test_dataset)
# 假设 test_data 包含您的测试数据，是一个列表

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
number2label = {0: 'Black-grass',
                1: 'Charlock',
                2: 'Cleavers',
                3: 'Common Chickweed',
                4: 'Common wheat',
                5: 'Fat Hen',
                6: 'Loose Silky-bent',
                7: 'Maize',
                8: 'Scentless Mayweed',
                9: 'Shepherds Purse',
                10: 'Small-flowered Cranesbill',
                11: 'Sugar beet'}
encoded_test_labels = []
out = []
for test_images in test_loader:
    test_images = test_images.to(device)
    test_images = test_images.float()
    y_pred = model(test_images).argmax(1)
    encoded_test_labels.extend(y_pred.tolist())

for i in encoded_test_labels:
    out.append(number2label[i])

output = pd.DataFrame({'ID': testdata, 'Category': out})  # 输出成一个dataframe
print(output)
print("end")
output.to_csv('submission2.csv', index=False, sep=',')  # 输出为csv文件
