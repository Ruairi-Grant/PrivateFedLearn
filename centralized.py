"""Script to train a model, with all of the data centralized"""

# General Utility imports
import os
import math
from pathlib import Path
import numpy as np

# Tensorflow model import
import tensorflow as tf
from keras import layers as tfkl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# Tensorflow Privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import dp_accounting

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Custom modules
from MySqueezeNet import SqueezeNet
import common

# Defing Hyperparamaters
EPOCHS = 70
BATCH_SIZE = 50
SEED = 42
IMAGE_SIZE = [265, 265]
DATA_DIR = Path(os.path.join("Datasets", "aptos2019-blindness-detection", "train"))
RESULTS_DIR = os.path.join("Results", "Centralized_Private_DR")


NOISE_MULTIPLIER = 0.3
DIFFERENTIAL_PRIVACY = True
L2_NORM_CLIP = 2
LEARNING_RATE = 0.02
MICROBATCHES = 10

tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
tf.config.experimental.enable_op_determinism()  # ensures all opereations are deterministic, maybe not nessessary without GPU


# TODO: what does this do
def compute_epsilon(epochs, num_data, batch_size):
    """Computes epsilon value for given hyperparameters."""
    steps = epochs * math.ceil(num_data / batch_size)
    if NOISE_MULTIPLIER == 0.0:
        return float("inf")
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    # TODO:does this assume that i used SGD
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = batch_size / num_data
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability, dp_accounting.GaussianDpEvent(NOISE_MULTIPLIER)
        ),
        steps,
    )

    accountant.compose(event)

    # TODO: find paramater for delta
    return accountant.get_epsilon(target_delta=1e-4)


# create train and test datasets
image_count = len(list(DATA_DIR.glob("*/*.png")))

# Create a dataset from the filtered file paths
list_ds = tf.data.Dataset.list_files(str(DATA_DIR / "*/*"), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in DATA_DIR.glob("*")]))

# split the dataset into training and validation sets
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(f"Training data size: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Validation data size: {tf.data.experimental.cardinality(val_ds).numpy()}")

# map the image paths to the images and labels
train_ds = train_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))
val_ds = val_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))

train_ds = common.configure_for_performance(train_ds, BATCH_SIZE)
val_ds = common.configure_for_performance(val_ds, BATCH_SIZE)

print(
    f"Training class distribution: {common.get_class_count(len(class_names), train_ds )}"
)
print(
    f"Validation class distribution: {common.get_class_count(len(class_names), val_ds )}"
)

# Data augmentation
data_prep = tf.keras.Sequential([tfkl.Rescaling(1.0 / 255), tfkl.CenterCrop(224, 224)])

data_augmentation = tf.keras.Sequential([data_prep, tfkl.RandomFlip("horizontal")])


# prepare the datasets
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
val_ds = val_ds.map(lambda x, y: (data_prep(x), y))
# test_ds = test_ds.map(lambda x, y: (data_prep(x), y))

print(
    f"Post processing training class distribution: {common.get_class_count(len(class_names), train_ds )}"
)
print(
    f"Post processing Validation class distribution: {common.get_class_count(len(class_names), val_ds )}"
)

# create the model
model = SqueezeNet(include_top=False, input_shape=(224, 224, 3))

# TODO: try changing back to adam
if DIFFERENTIAL_PRIVACY:
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=L2_NORM_CLIP,
        noise_multiplier=NOISE_MULTIPLIER,
        num_microbatches=MICROBATCHES,
    )
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.losses.Reduction.NONE
    )
else:
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=LEARNING_RATE,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy"],
)

model.summary()

# simple early stopping
# early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

# checkpoint = ModelCheckpoint(
#    "best_model.h5", monitor="val_loss", mode="min", verbose=1, save_best_only=True
# )

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    # callbacks=[early_stopping, checkpoint],
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(loss))

# check if the results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# PLot the dataset and save it
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(epochs_range, acc, label="Training Accuracy")
ax1.plot(epochs_range, val_acc, label="Validation Accuracy")
ax1.legend(loc="lower right")

fig1.savefig(os.path.join(RESULTS_DIR, "TrainingValidationAccuracy"))

# PLot the dataset and save it
fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.plot(epochs_range, loss, label="Training Loss")
ax2.plot(epochs_range, val_loss, label="Validation Loss")
ax2.legend(loc="lower right")
fig2.savefig(os.path.join(RESULTS_DIR, "TrainingValidationLoss"))

# Load the best model
#best_model = load_model(
#    "best_model.h5"
#)  # TODO: Error here saying unknown optimizer, see github issue https://github.com/keras-team/tf-keras/issues/297

# Evaluate the model
common.evaluate_model(model, train_ds, os.path.join(RESULTS_DIR, "Train"))
common.evaluate_model(model, val_ds, os.path.join(RESULTS_DIR, "Validation"))

# TODO: make this accurate for my case
# Compute the privacy budget expended.
if DIFFERENTIAL_PRIVACY:
    # eps = compute_epsilon(EPOCHS * 60000 // BATCH_SIZE)
    eps = compute_epsilon(
        len(loss), tf.data.experimental.cardinality(train_ds).numpy(), BATCH_SIZE
    )
    print(f"For delta=1e-5, the current epsilon is: {eps}")
else:
    print("Trained with vanilla non-private SGD optimizer")
