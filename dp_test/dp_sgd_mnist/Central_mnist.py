import os
import math

# SKlearn model evaluation
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer
import dp_accounting


from typing import List, Tuple

import numpy as np

#from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
#from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]



def compute_epsilon(
    epochs: int, num_train_examples: int, batch_size: int, noise_multiplier: float
) -> float:
    """Computes epsilon value for given hyperparameters.

    Based on
    github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_keras.py
    """
    if noise_multiplier == 0.0:
        return float("inf")
    steps = epochs * num_train_examples // batch_size
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    sampling_probability = batch_size / num_train_examples
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )

    accountant.compose(event)

    # Delta is set to approximate 1 / (number of training points).
    return accountant.get_epsilon(target_delta=1e-4)


def create_cnn_model() -> tf.keras.Model:
    """Returns a sequential keras CNN Model."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16,
                8,
                strides=2,
                padding="same",
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            tf.keras.layers.MaxPool2D(2, 1),
            tf.keras.layers.Conv2D(
                32, 4, strides=2, padding="valid", activation="relu"
            ),
            tf.keras.layers.MaxPool2D(2, 1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


def preprocess(X: np.ndarray, y: np.ndarray) -> XY:
    """Basic preprocessing for MNIST dataset."""
    X = np.array(X, dtype=np.float32) / 255
    X = X.reshape((X.shape[0], 28, 28, 1))

    y = np.array(y, dtype=np.int32)
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    return X, y


def create_partitions(source_dataset: XY, num_partitions: int) -> XYList:
    """Create partitioned version of a source dataset."""
    X, y = source_dataset
    X, y = shuffle(X, y)
    X, y = preprocess(X, y)
    xy_partitions = partition(X, y, num_partitions)
    return xy_partitions


def load(
    num_partitions: int,
) -> PartitionedDataset:
    """Create partitioned version of MNIST."""
    xy_train, xy_test = tf.keras.datasets.mnist.load_data()
    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)
    return list(zip(xy_train_partitions, xy_test_partitions))


RESULTS_DIR = os.path.join("Results", "Centralized_Private_DR")

BATCH_SIZE = 32
LOCAL_EPOCHS = 30
LEARNING_RATE = 0.1

DPSGD = True
L2_NORM_CLIP = 1.0
NOISE_MULTIPLIER = 1.1
MICROBATCHES = 32

def evaluate_model(eval_model, dataset, dir_path):
    """Function to evaluate the model and save the confusion matrix and classification report"""
    _, eval_acc = eval_model.evaluate(dataset, verbose=1)
    print("\nTrain accuracy:", eval_acc)
    # Clear the current matplotlib figure
    plt.clf()
    # Get the true labels and predicted labels
    true_labels = []
    predicted_labels = []

    for images, labels in dataset:
        predictions = eval_model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))
        true_labels.extend(labels.numpy())

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(dir_path + "/confusion_matrix.png")

    # Create the classification report
    report = classification_report(true_labels, predicted_labels)

    # check if the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(dir_path + "/classification_report.txt", "w") as file:
        file.write(report)


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

def main():
    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load(1)[0]

    model = create_cnn_model()

    if DPSGD and x_train.shape[0] % BATCH_SIZE != 0:
        drop_num = x_train.shape[0] % BATCH_SIZE
        x_train = x_train[:-drop_num]
        y_train = y_train[:-drop_num]

    if DPSGD:
        if BATCH_SIZE % MICROBATCHES != 0:
            raise ValueError(
                "Number of microbatches should divide evenly batch_size"
            )
        optimizer = VectorizedDPKerasSGDOptimizer(
            l2_norm_clip=L2_NORM_CLIP,
            noise_multiplier=NOISE_MULTIPLIER,
            num_microbatches=MICROBATCHES,
            learning_rate=LEARNING_RATE,
        )
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE
        )
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_train, y_train),
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

    # Evaluate the model
    evaluate_model(model, (x_train, y_train), os.path.join(RESULTS_DIR, "Train"))
    evaluate_model(model, (x_test, y_test), os.path.join(RESULTS_DIR, "Validation"))

    # TODO: make this accurate for my case
    # Compute the privacy budget expended.
    if DPSGD:
        # eps = compute_epsilon(EPOCHS * 60000 // BATCH_SIZE)
        eps = compute_epsilon(
            len(loss), tf.data.experimental.cardinality(train_ds).numpy(), BATCH_SIZE
        )
        print(f"For delta=1e-5, the current epsilon is: {eps}")
    else:
        print("Trained with vanilla non-private SGD optimizer")
