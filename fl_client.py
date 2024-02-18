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
SEED = 42
BATCH_SIZE = 50
LOCAL_EPOCHS = 50  # TODO: Figure out how to get this from the server


def split_data():
    # Assuming your dataframe is named df and the category column is named 'category'
    df = pd.read_csv(
        "C:\\git_repos\\Thesis\\Datasets\\aptos2019-blindness-detection\\train.csv"
    )

    # Step 1: Group by the diagnosis column
    grouped = df.groupby("diagnosis")

    # Step 2: Sort each group by the diagnosis column
    sorted_groups = {k: v.sort_values("diagnosis") for k, v in grouped}

    # Step 3: Split each group into four groups with slightly unequal sizes
    num_subgroups = 4
    subgroups = {k: [] for k in sorted_groups.keys()}
    for k, v in sorted_groups.items():
        group_size = len(v)
        subgroup_base_size = group_size // num_subgroups
        remainder = group_size % num_subgroups
        start = 0
        for i in range(num_subgroups):
            subgroup_size = subgroup_base_size + (1 if i < remainder else 0)
            subgroup = v.iloc[start : start + subgroup_size]
            subgroups[k].append(subgroup)
            start += subgroup_size

    # Step 4: Combine subgroups from different diagnoses into the final groups
    final_groups = []
    for i in range(num_subgroups):
        final_group = pd.concat([subgroups[k][i] for k in sorted_groups.keys()])
        final_groups.append(final_group)

    return final_groups


def prepare_dataset(data_df):
    """Download and partitions the CIFAR-10/MNIST dataset."""
    # TODO: I need to load the data here and then partitian it
    # TODO: This is also where data augmentation needs to happen

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

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(f"Training data size: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation data size: {tf.data.experimental.cardinality(val_ds).numpy()}")

    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [265, 265])

    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    def get_class_count(num_classes, dataset):
        count = np.zeros(num_classes, dtype=np.int32)
        for _, labels in dataset:
            y, _, c = tf.unique_with_counts(labels)
            count[y.numpy()] += c.numpy()
        return count

    train_ds = train_ds.map(process_path)
    val_ds = val_ds.map(process_path)

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(BATCH_SIZE)
        # ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

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
