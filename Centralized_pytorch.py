import os
import numpy as np
import pandas as pd
import urllib.request
import zipfile
import gdown
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

from torch.utils.data import Dataset, TensorDataset, DataLoader

# load dataset and user groups
# train_dataset, test_dataset, user_groups = get_dataset(args)

class DRDataset(Dataset):
    def __init__(self, data_label, data_dir, transform):
        super().__init__()
        self.data_label = data_label
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, index):
        img_name = self.data_label.id_code[index] + '.png'
        label = self.data_label.diagnosis[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        image = self.transform(image)
        return image, label

# TODO: have a better way of setting this
DATA_LOADED = True
NUMPY_DIR = "NumpyData"
DEVICE = 'cpu'
BATCH_SIZE = 50
LR = 0.01
MAX_EPOCH = 100
# work out the story with iid
if DATA_LOADED:
    _x = torch.Tensor(np.load(os.path.join(NUMPY_DIR,"dr_train_images.npy")))
    _y = torch.Tensor(np.load(os.path.join(NUMPY_DIR,"dr_train_labels.npy"))).long()
    train_dataset = TensorDataset(_x,_y)
    _x = torch.Tensor(np.load(os.path.join(NUMPY_DIR,"dr_test_images.npy")))
    _y = torch.Tensor(np.load(os.path.join(NUMPY_DIR,"dr_test_labels.npy"))).long()
    test_dataset = TensorDataset(_x,_y)            

else:
    DATA_DIR = '../data/diabetic_retinopathy/'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    #download ZIP, unzip it, delete zip file
    DATA_URL = "https://drive.google.com/uc?id=1G-4UhPKiQY3NxQtZiWuOkdocDTW6Bw0u"
    zip_path = DATA_DIR + 'images.zip'
    gdown.download(DATA_URL, zip_path, quiet=False)
    print("Extracting...!")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extracted!")
    os.remove(zip_path)

    #download train and test dataframes
    test_csv_link = 'https://drive.google.com/uc?id=1dmeYLURzEvx962th4lAxaVN3r6nlhTjS'
    train_csv_link = 'https://drive.google.com/uc?id=1SMb9CRHjB6UH2WnTZDFVSgpA6_nh75qN'
    test_csv_path = DATA_DIR + 'test_set.csv'
    train_csv_path = DATA_DIR + 'train_set.csv'
    urllib.request.urlretrieve(train_csv_link, train_csv_path)
    urllib.request.urlretrieve(test_csv_link, test_csv_path)
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    #create train and test datasets
    apply_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(265),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image_directory = DATA_DIR + 'images/'
    train_dataset = DRDataset(data_label = df_train, data_dir = image_directory,
                                transform = apply_transform)
    test_dataset = DRDataset(data_label = df_test, data_dir = image_directory,
                                transform = apply_transform)

    print("train_dataset size:", len(train_dataset))
    print("test_dataset size:", len(test_dataset))
    # print("data shape:", train_dataset[0][0].shape)
    print("train")
    dr_images = []
    dr_labels = []
    for i in range(len(train_dataset)):        
        _image, _label = train_dataset[i]
        dr_images.append(_image.numpy())
        dr_labels.append(_label)
        print("  ", i, end="\r")
    print("")
    dr_images = np.array(dr_images)
    dr_labels = np.array(dr_labels)
    np.save(os.path.join(NUMPY_DIR,"dr_train_images.npy"), dr_images)
    np.save(os.path.join(NUMPY_DIR,"dr_train_labels.npy"), dr_labels)
    print("test")
    dr_images = []
    dr_labels = []
    for i in range(len(test_dataset)):        
        _image, _label = test_dataset[i]
        dr_images.append(_image.numpy())
        dr_labels.append(_label)
        print("  ", i, end="\r")
    print("")
    dr_images = np.array(dr_images)
    dr_labels = np.array(dr_labels)
    np.save(os.path.join(NUMPY_DIR,"dr_test_images.npy"), dr_images)
    np.save(os.path.join(NUMPY_DIR,"dr_test_labels.npy"), dr_labels)

# get data
trainloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, drop_last=True) 

# Split the dataset into training and validation sets
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

global_model = models.squeezenet1_1(pretrained=True)           
global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
global_model.num_classes = 5
global_model.to(DEVICE)
summary(global_model, input_size=(3, 32, 32), device=DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(global_model.parameters(), lr=LR)  


# Early stopping parameters
patience = 3
best_val_loss = float('inf')
counter = 0

# Training loop
for epoch in range(MAX_EPOCH):  # You can adjust the number of epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = global_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
