import os

import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
    VectorizedDPKerasSGDOptimizer,
)

import flwr as fl

from typing import List, Tuple

import numpy as np
import tensorflow as tf

# from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
# from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
import dp_accounting

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


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# global for tracking privacy
PRIVACY_LOSS = 0


# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, batch_size, local_epochs, dpsgd, l2_norm_clip, noise_multiplier, microbatches, learning_rate):
        # small model for MNIST
        self.model = create_cnn_model()
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.dpsgd = dpsgd

        if dpsgd:
            self.noise_multiplier = noise_multiplier
            if batch_size % microbatches != 0:
                raise ValueError(
                    "Number of microbatches should divide evenly batch_size"
                )
            optimizer = VectorizedDPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=learning_rate,
            )
            # Compute vector of per-example loss rather than its mean over a minibatch.
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction=tf.losses.Reduction.NONE
            )
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile model with Keras
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("Client sampled for fit()")
        # Update local model parameters
        global PRIVACY_LOSS
        if self.dpsgd:
            privacy_spent = compute_epsilon(
                self.local_epochs,
                len(self.x_train),
                self.batch_size,
                self.noise_multiplier,
            )
            PRIVACY_LOSS += privacy_spent

        print("Privacy Loss: ", PRIVACY_LOSS)
        self.model.set_weights(parameters)
        # Train the model
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
        )

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    
    num_clients = 2
    partition = 0
    local_epochs = 3
    batch_size = 32
    learning_rate = 0.15
    dpsgd = True
    l2_norm_clip = 1.0
    noise_multiplier = 1.1
    microbatches = 32
    
    
    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load(num_clients)[partition]

    # drop samples to form exact batches for dpsgd
    # this is necessary since dpsgd is sensitive to uneven batches
    # due to microbatching
    if dpsgd and x_train.shape[0] % batch_size != 0:
        drop_num = x_train.shape[0] % batch_size
        x_train = x_train[:-drop_num]
        y_train = y_train[:-drop_num]

    # Start Flower client
    client = MnistClient(x_train, y_train, x_test, y_test,batch_size=batch_size, local_epochs=local_epochs, dpsgd=dpsgd, l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, microbatches=microbatches, learning_rate=learning_rate)
    fl.client.start_numpy_client(server_address="192.168.0.10:8080", client=client)
    if dpsgd:
        print("Privacy Loss: ", PRIVACY_LOSS)

