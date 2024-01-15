# General Utility functions
import os
import shutil
import pandas as pd
from PIL import Image 
import numpy as np
from pathlib import Path
import matplotlib as plt

# Tensorflow model import
import tensorflow  as tf
from tensorflow.keras import layers as tfkl
from MySqueezeNet import SqueezeNet

from keras.applications.resnet50 import preprocess_input, decode_predictions


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

# Defing Hyperparamaters
BATCH_SIZE = 50
SEED = 42 



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

DATA_DIR = r'Datasets/aptos2019-blindness-detection'

train_csv_path = DATA_DIR + '/train.csv'

df_train = pd.read_csv(train_csv_path)

# TODO: See if this can be more efficient, ie. check if it has already been done
# split all the images into the foldair according to their label
for diagnosis, group in df_train.groupby('diagnosis'):
    path_name = Path(DATA_DIR).joinpath('train', 'class_' +  str(diagnosis))
    create_folder(path_name)
    class_names = list(group['id_code'])
    class_paths = [Path(DATA_DIR).joinpath('train_images', name + '.png') for name in class_names]
    new_paths = [path_name.joinpath(name + '.png') for name in class_names]
    for src, dst in zip(class_paths, new_paths):
        shutil.copy(src, dst)




#create train and test datasets
        
train_ds = tf.keras.utils.image_dataset_from_directory(Path(DATA_DIR).joinpath('train'),
                                                    validation_split=0.3,
                                                    subset="training",
                                                    color_mode='rgb',
                                                    seed=SEED,
                                                    image_size=(512, 512),  # This resizes the image, using bilinear transformation
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True
                                                    )

val_ds = tf.keras.utils.image_dataset_from_directory(Path(DATA_DIR).joinpath('train'),
                                                    validation_split=0.3,
                                                    subset="validation",
                                                    color_mode='rgb',
                                                    seed=SEED,
                                                    image_size=(512, 512),  # This resizes the image, using bilinear transformation
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True
                                                    )


# could probably work batch count out but this is clearer
# split validation into test and val
# TODO:print the size of these datasets
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take((2*val_batches) // 3)
val_ds = val_ds.skip((2*val_batches) // 3)


data_prep = tf.keras.Sequential([
        tfkl.Rescaling(1./255),
        tfkl.Resizing(265, 265),
        tfkl.CenterCrop(224, 224)
])


data_augmentation = tf.keras.Sequential([
    data_prep,
    tfkl.RandomFlip("horizontal")
])


#prepare the datasets
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
val_ds = val_ds.map(lambda x, y: (data_prep(x), y))
test_ds = test_ds.map(lambda x, y: (data_prep(x), y))
 
# create the model
model = SqueezeNet(include_top=False, input_shape=(224, 224, 3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

epochs=100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
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