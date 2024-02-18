'''Script to train a model, with all of the data centralized'''

# General Utility functions
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tensorflow model import
import tensorflow as tf
from keras import layers as tfkl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from MySqueezeNet import SqueezeNet
import common

# Defing Hyperparamaters
EPOCHS = 40
BATCH_SIZE = 50
SEED = 42
IMAGE_SIZE = [265, 265]
DATA_DIR = Path("Datasets\\aptos2019-blindness-detection\\train")

# create train and test datasets
image_count = len(list(DATA_DIR.glob("*/*.png")))

# Create a dataset from the filtered file paths
list_ds = tf.data.Dataset.list_files(str(DATA_DIR / "*/*"), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in DATA_DIR.glob("*")]))

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(f"Training data size: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Validation data size: {tf.data.experimental.cardinality(val_ds).numpy()}")

def get_class_count(num_classes, dataset):
    '''Function to get the class count of the dataset'''
    count = np.zeros(num_classes, dtype=np.int32)
    for _, labels in dataset:
        y, _, c = tf.unique_with_counts(labels)
        count[y.numpy()] += c.numpy()
    return count


train_ds = train_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))
val_ds = val_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))

train_ds = common.configure_for_performance(train_ds, BATCH_SIZE)
val_ds = common.configure_for_performance(val_ds, BATCH_SIZE)

print(f"Training class distribution: {get_class_count(len(class_names), train_ds )}")
print(f"Validation class distribution: {get_class_count(len(class_names), val_ds )}")

data_prep = tf.keras.Sequential(
    [
        # tfkl.Rescaling(1./255),
        tfkl.CenterCrop(224, 224)
    ]
)


data_augmentation = tf.keras.Sequential([data_prep, tfkl.RandomFlip("horizontal")])


# prepare the datasets
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
val_ds = val_ds.map(lambda x, y: (data_prep(x), y))
# test_ds = test_ds.map(lambda x, y: (data_prep(x), y))

print(
    f"Post processing training class distribution: {get_class_count(len(class_names), train_ds )}"
)
print(
    f"Post processing Validation class distribution: {get_class_count(len(class_names), val_ds )}"
)

# create the model
model = SqueezeNet(include_top=False, input_shape=(224, 224, 3))

# tf.keras.optimizers.experimental.SGD(momentum=0.9)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()


# simple early stopping
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

checkpoint = ModelCheckpoint(
    "best_model.h5", monitor="val_accuracy", mode="max", verbose=1, save_best_only=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint],
)

best_model = load_model("best_model.h5")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(loss))

# PLot the dataset and save it
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(epochs_range, acc, label="Training Accuracy")
ax1.plot(epochs_range, val_acc, label="Validation Accuracy")
ax1.legend(loc="lower right")

fig1.savefig("Figures\\TrainingValidationAccuracy")

# PLot the dataset and save it
fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.plot(epochs_range, loss, label="Training Loss")
ax2.plot(epochs_range, val_loss, label="Validation Loss")
ax2.legend(loc="lower right")
fig2.savefig("Figures\\TrainingValidationLoss")

test_loss, test_acc = model.evaluate(train_ds, verbose=2)

print("\nTest accuracy:", test_acc)
