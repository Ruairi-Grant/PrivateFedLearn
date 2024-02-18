import os
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

import flwr as fl
from flwr.common import Config, NDArrays, Scalar
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras import layers as tfkl
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU

from MySqueezeNet import SqueezeNet
import common

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)


warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 4
DATA_DIR = Path("Datasets\\aptos2019-blindness-detection\\train")
DATA_DF_DIR = Path("Datasets\\aptos2019-blindness-detection\\train.csv")
SEED = 42
BATCH_SIZE = 50
LOCAL_EPOCHS = 50  # TODO: Figure out how to get this from the server
IMAGE_SIZE = [265, 265]


def split_data():
    """Function to split the data into four groups based on the diagnosis column

    Returns:
        list: A list of four pandas DataFrames, each representing a group of data.
    """

    # Load the data from the CSV file
    data_df = pd.read_csv(DATA_DF_DIR)

    # Group by the diagnosis column
    data_df = data_df.groupby("diagnosis")

    # Sort each group by the diagnosis column
    sorted_groups = {k: v.sort_values("diagnosis") for k, v in data_df}

    # Split each group into four groups that are as close in size as possible
    num_subgroups = 4
    subgroups = {k: [] for k in sorted_groups.keys()}
    for k, v in sorted_groups.items():
        group_size = len(v)
        subgroup_base_size = group_size // num_subgroups
        remainder = group_size % num_subgroups
        start = 0
        for i in range(num_subgroups):
            subgroup_size = subgroup_base_size + (1 if i < remainder else 0)
            subgroups[k].append(v.iloc[start : start + subgroup_size])
            start += subgroup_size

    # Step 4: Combine subgroups from different diagnoses into the final groups
    final_groups = []
    for i in range(num_subgroups):
        final_group = pd.concat([subgroups[k][i] for k in sorted_groups.keys()])
        final_groups.append(final_group)

    return final_groups


def prepare_dataset(data_df):
    """Download and partitions the CIFAR-10/MNIST dataset."""
    # Define your list of allowed filenames
    allowed_files_set = set(data_df["id_code"] + ".png")

    # Filter files in the data directory based on the allowed filenames
    filtered_files = [
        str(file_path)
        for file_path in DATA_DIR.glob("*/*")
        if file_path.name in allowed_files_set
    ]

    image_count = len(filtered_files)

    # Create a dataset from the filtered file paths
    list_ds = tf.data.Dataset.from_tensor_slices(filtered_files)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted([item.name for item in DATA_DIR.glob("*")]))

    # split the dataset into training and validation sets
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(f"Training data size: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation data size: {tf.data.experimental.cardinality(val_ds).numpy()}")

    def get_class_count(num_classes, dataset):
        count = np.zeros(num_classes, dtype=np.int32)
        for _, labels in dataset:
            y, _, c = tf.unique_with_counts(labels)
            count[y.numpy()] += c.numpy()
        return count

    # map the image paths to the images and labels
    train_ds = train_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))
    val_ds = val_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))

    # configure the datasets for performance
    train_ds = common.configure_for_performance(train_ds, BATCH_SIZE)
    val_ds = common.configure_for_performance(val_ds, BATCH_SIZE)

    data_prep = tf.keras.Sequential(
        [
            # tfkl.Rescaling(1./255),
            tfkl.CenterCrop(224, 224)
        ]
    )

    data_augmentation = tf.keras.Sequential([data_prep, tfkl.RandomFlip("horizontal")])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
    val_ds = val_ds.map(lambda x, y: (data_prep(x), y))
    print(
        f"Training class distribution: {get_class_count(len(class_names), train_ds )}"
    )
    print(
        f"Validation class distribution: {get_class_count(len(class_names), val_ds )}"
    )

    return (train_ds, val_ds)


class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that uses SqueeseNet."""

    def __init__(self, trainset, valset):
        self.x_train_ds = trainset
        self.x_val_ds = valset
        # Instantiate model
        self.model = SqueezeNet(include_top=False, input_shape=(224, 224, 3))
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def get_parameters(self, config) -> NDArrays:
        weights = self.model.get_weights()
        # weights = 1
        # print(weights)
        return weights

    def set_parameters(self, params):
        # print(params)
        self.model.set_weights(params)

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Set hyperparameters from config sent by server/strategy
        batch, epochs = config["batch_size"], config["epochs"]
        # train
        self.model.fit(self.x_train_ds, epochs=epochs, batch_size=batch)
        w = self.get_parameters({})
        model_len = int(self.x_train_ds.cardinality().numpy())
        result = {}
        # assert isinstance(w, np.ndarray)
        assert isinstance(model_len, int)
        # assert isinstance(result, dict)
        return w, model_len, result

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_val_ds)
        return loss, int(self.x_val_ds.cardinality().numpy()), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    # Download CIFAR-10 dataset and partition it
    data_split = split_data()
    trainset, valset = prepare_dataset(data_split[args.cid])

    # Start Flower client setting its associated data partition
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(trainset=trainset, valset=valset),
    )


if __name__ == "__main__":
    main()
