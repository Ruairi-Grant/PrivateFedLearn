"""Script to train a model, with all of the data centralized"""

# General Utility functions
from pathlib import Path
import matplotlib.pyplot as plt

# Tensorflow model import
import tensorflow as tf
from keras import layers as tfkl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from MySqueezeNet import SqueezeNet

# Defing Hyperparamaters
EPOCHS = 100
BATCH_SIZE = 50
SEED = 42
DATA_DIR = "data/diabetic_retinopathy/"

# create train and test datasets

train_ds = tf.keras.utils.image_dataset_from_directory(
    Path(DATA_DIR).joinpath("train"),
    validation_split=0.1,
    subset="training",
    color_mode="rgb",
    seed=SEED,
    image_size=(265, 265),  # This resizes the image, using bilinear transformation
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    Path(DATA_DIR).joinpath("train"),
    validation_split=0.1,
    subset="validation",
    color_mode="rgb",
    seed=SEED,
    image_size=(265, 265),  # This resizes the image, using bilinear transformation
    batch_size=BATCH_SIZE,
    shuffle=True,
)


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
