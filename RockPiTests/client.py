import os
import argparse
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
    VectorizedDPKerasSGDOptimizer,
)

import flwr as fl

from typing import List, Tuple

import numpy as np
import tensorflow as tf

import common

# from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
# from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
#import dp_accounting

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# global for tracking privacy
PRIVACY_LOSS = 0


# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size,
        local_epochs,
        dpsgd,
        l2_norm_clip,
        noise_multiplier,
        microbatches,
        learning_rate,
    ):
        # small model for MNIST
        self.model = common.create_cnn_model()
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
            privacy_spent = common.compute_epsilon(
                self.local_epochs,
                len(self.x_train),
                self.batch_size,
                self.noise_multiplier,
            )
            PRIVACY_LOSS += privacy_spent

        print("Privacy Loss: ", PRIVACY_LOSS)
        self.model.set_weights(parameters)
        # Train the model
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_test, self.y_test),
        )

        # Flower doesn't allow the elemetns of the dict to be anything but a scalar, so we can only return the last element of the history
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            #"privacy_loss": PRIVACY_LOSS,
        }

        return self.model.get_weights(), len(self.x_train), results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"accuracy": accuracy}


def main(dpsgd: bool, server_address: str, partition: int, num_clients: int) -> None:

    local_epochs = 3
    batch_size = 32
    learning_rate = 0.15
    l2_norm_clip = 1.0
    noise_multiplier = 1.1
    microbatches = 32

    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = common.load(num_clients)[partition]

    # drop samples to form exact batches for dpsgd
    # this is necessary since dpsgd is sensitive to uneven batches
    # due to microbatching
    if dpsgd and x_train.shape[0] % batch_size != 0:
        drop_num = x_train.shape[0] % batch_size
        x_train = x_train[:-drop_num]
        y_train = y_train[:-drop_num]

    # Start Flower client
    client = MnistClient(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=batch_size,
        local_epochs=local_epochs,
        dpsgd=dpsgd,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        microbatches=microbatches,
        learning_rate=learning_rate,
    )
    fl.client.start_numpy_client(server_address=server_address, client=client)
    if dpsgd:
        print("Privacy Loss: ", PRIVACY_LOSS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--dpsgd",
        type=bool,
        default=False,
        required=False,
        help="Data Partion to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="gRPC server address (default '0.0.0.0:8080')",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default="0",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default="3",
    )

    args = parser.parse_args()

    main(args.dpsgd, args.server_address, args.partition, args.num_clients)
