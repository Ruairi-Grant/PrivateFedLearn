# General Utility functions
import os
import shutil
import pandas as pd
from PIL import Image 
import numpy as np
from pathlib import Path

# Tensorflow model imports
from tensorflow import keras
from MySqueezeNet import SqueezeNet
from keras.layers import Convolution2D
from keras.applications.resnet50 import preprocess_input, decode_predictions


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

# Defing Hyperparamaters
BATCH_SIZE = 50



def create_folder(path_name):
    if not path_name.is_dir():
        path_name.mkdir(parents=True)

def copy_file(src,dst):
    if dst.is_file():
        # if file exists then write behavior here.
        return 1
    else:
        shutil.copy(src, dst)

# This bit may be useful for loading the data faster 

"""
_x = torch.Tensor(np.load("dr_train_images.npy"))
_y = torch.Tensor(np.load("dr_train_labels.npy")).long()
train_dataset = TensorDataset(_x,_y)
_x = torch.Tensor(np.load("dr_test_images.npy"))
_y = torch.Tensor(np.load("dr_test_labels.npy")).long()
test_dataset = TensorDataset(_x,_y)            
"""

DATA_DIR = r'Datasets/aptos2019-blindness-detection'


#download ZIP, unzip it, delete zip file

#download train and test dataframes
#test_csv_path = DATA_DIR + '/test.csv'
train_csv_path = DATA_DIR + '/train.csv'

df_train = pd.read_csv(train_csv_path)
#df_test = pd.read_csv(test_csv_path)

for diagnosis, group in df_train.groupby('diagnosis'):
    path_name = Path(DATA_DIR).joinpath('train', 'class_' +  str(diagnosis))
    create_folder(path_name)
    class_names = list(group['id_code'])
    class_paths = [Path(DATA_DIR).joinpath('train_images', name + '.png') for name in class_names]
    new_paths = [path_name.joinpath(name + '.png') for name in class_names]
    for src, dst in zip(class_paths, new_paths):
        shutil.copy(src, dst)


# split all the images into the foldair according to their label

#create train and test datasets
# PyTorch equivlient

"""
apply_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(265),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""
"""
train_ds = keras.utils.image_dataset_from_directory(data_dir,
                                                    validation_split=0.2,
                                                    subset="training",
                                                    color_mode='rgb',
                                                    seed=42,
                                                    image_size=(256, 256),
                                                    batch_size=BATCH_SIZE
                                                    )

val_ds = keras.utils.image_dataset_from_directory(data_dir,
                                                    validation_split=0.2,
                                                    subset="validation",
                                                    color_mode='rgb',
                                                    seed=42,
                                                    image_size=(256, 256),
                                                    batch_size=BATCH_SIZE
                                                    )

test_ds = keras.utils.image_dataset_from_directory(data_dir,
                                                    color_mode='rgb',
                                                    seed=42,
                                                    image_size=(256, 256),
                                                    batch_size=BATCH_SIZE
                                                    )
"""
#image_directory = data_dir + 'images/'
#train_dataset = DRDataset(data_label = df_train, data_dir = image_directory,transform = apply_transform)
#test_dataset = DRDataset(data_label = df_test, data_dir = image_directory,transform = apply_transform)   
"""
#return train_dataset, test_dataset, user_groups
# load dataset and user groups
train_dataset, test_dataset, user_groups = get_dataset(args)

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
np.save("dr_train_images.npy", dr_images)
np.save("dr_train_labels.npy", dr_labels)
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
np.save("dr_test_images.npy", dr_images)
np.save("dr_test_labels.npy", dr_labels)

print("Done!")

model = SqueezeNet(include_top=False, input_shape=(512, 512, 3))


model.summary()
"""
"""
global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
global_model.num_classes = 5
img_path = 'elephant.jpg'
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
"""