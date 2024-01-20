import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import os
from torchvision import models
from torch import nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torchvision
# 自定义数据集类

torch.manual_seed(22)

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]  # 假设第一列是图片文件名
        img_name = str(img_name)

        img_path = os.path.join(self.root_dir, img_name+".jpg")
        image = Image.open(img_path).convert('RGB')

        # 假设标签在第二列
        label = int(self.df.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)

        return image, label

# 读取CSV文件
csv_path = 'D:/Users/dk/Desktop/kaggle/grade/train.csv'
root_dir = 'D:/Users/dk/Desktop/kaggle/grade/train'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



# 定义转换
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])


# 创建数据集
dataset = CustomDataset(csv_file='D:/Users/dk/Desktop/kaggle/grade/train.csv', root_dir='D:/Users/dk/Desktop/kaggle/grade/train', transform=transform)
#test_dataset = CustomDataset(csv_file='D:/Users/dk/Desktop/kaggle/grade/train.csv', root_dir='D:/Users/dk/Desktop/kaggle/grade/train', transform=transform)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# 创建 DataLoader
data_loader = DataLoader(dataset,batch_size=32,shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


resnet50 = models.resnet50(pretrained=True)
print(sum(1 for _ in resnet50.parameters()))
num_ftrs = resnet50.fc.in_features
#resnet50.fc = nn.Sequential(nn.Linear(num_ftrs,5))
#resnet50.load_state_dict(torch.load('C:/Users/dk/PycharmProjects/Last/pretrainmodel.pth'))

resnet50.fc = nn.Sequential(nn.Linear(num_ftrs,1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50
model = model.to(device)
print(model)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10  # 训练轮次
train_loss_list = []  # 损失集合
train_acc = 0  # 训练准确率
count1 = 1000  # dataset的数量

model.train()
for epoch in range(num_epochs):
    model.train()
    print(epoch+1)
    poch_acc = 0
    train_loss = 0
    epoch_acc = 0
    for images, labels in train_loader:
        labels = labels.type(torch.float)
        images, labels = images.to(device), labels.to(device)
        images = images.float()
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
        epoch_acc += (y_pred.argmax(1) == labels).sum().item()
    print('Epoch: {} - Loss: {:.6f}, Training Acc.: {:.3f}'.format(epoch + 1, train_loss / train_size, epoch_acc / train_size))
    model.eval()
    epoch_acc1 = 0
    a1 = []
    a2 = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            y_pred1 = model(images)
            epoch_acc1 += (y_pred1.argmax(1) == labels).sum().item()
            # 计算准确率
            a = labels.to('cpu').numpy()
            a1.extend(a)
            b = y_pred1.argmax(1).to('cpu').numpy()
            a2.extend(b)
        print('Epoch: {}, Val Acc.: {:.3f}'.format(epoch + 1, epoch_acc1 / val_size))
        """
        cm = confusion_matrix(a1, a2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                    yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        """
path = "D:/Users/dk/Desktop/kaggle/grade"
image_size = 380
files = os.listdir(path + '/test')
files = sorted(files, key=lambda x: int(x.split('.')[0]))
test_dataset = list()

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
for i in files:
    image = Image.open(path + '/test/' + i)
    image_tensor = transform(image)
    test_dataset.append(image_tensor)
testdata = files
print(len(test_dataset))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
model.eval()
encoded_test_labels = []
for test_images in test_loader:
    test_images = test_images.to(device)
    test_images = test_images.float()
    y_pred = model(test_images).argmax(1)
    encoded_test_labels.extend(y_pred.tolist())

for i in encoded_test_labels:
    print(i)

sample = pd.read_csv("C:/Users/dk/PycharmProjects/Last/submission2.csv")
submission = pd.DataFrame({'Image': sample['Image'], 'Predict': encoded_test_labels})
submission.to_csv('C:/Users/dk/PycharmProjects/Last/submission2.csv', index=False)


