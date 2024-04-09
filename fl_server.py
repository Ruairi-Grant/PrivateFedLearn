"""Module to define the FlowerServer class and the main function to start the server"""

# General Utility imports
import os
import argparse
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Tensorflow and Keras imports
import tensorflow as tf
from tensorflow import keras
from keras import layers as tfkl

import flwr as fl
from flwr.common import Metrics

# Custom modules
from MySqueezeNet import SqueezeNet
import common


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.68.64:8080",
    help="gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--mnist",
    action="store_true",
    help="If you use Raspberry Pi Zero clients (which just have 512MB or RAM) use MNIST",
)

parser.add_argument(
    "--local_epochs",
    type=int,
    default=50,
    help="Local epochs done by clients (default: 50)",
)

parser.add_argument(
    "--local_batch",
    type=int,
    default=50,
    help="Local epochs done by clients (default: 50)",
)

# TODO: is there a better way to handle these
## Define default valued for the global variables
LOCAL_EPOCHS = 50
LOCAL_BATCH = 50
IMAGE_SIZE = [265, 265]

DATA_DIR = Path("Datasets\\aptos2019-blindness-detection\\train")
RESULTS_DIR = Path("Results\\FedLearn_DR_privacy")
VAL_DF_DIR = Path("Datasets\\aptos2019-blindness-detection\\split_val.csv")


def prepare_dataset_for_eval(data_df):
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
    eval_ds = tf.data.Dataset.from_tensor_slices(filtered_files)
    eval_ds = eval_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted([item.name for item in DATA_DIR.glob("*")]))

    print(f"Evaluation data size: {tf.data.experimental.cardinality(eval_ds).numpy()}")

    # map the image paths to the images and labels
    eval_ds = eval_ds.map(lambda x: common.process_path(x, class_names, IMAGE_SIZE))

    # configure the datasets for performance
    eval_ds = common.configure_for_performance(eval_ds, 1)

    data_prep = tf.keras.Sequential(
        [
            # tfkl.Rescaling(1./255),
            tfkl.CenterCrop(224, 224)
        ]
    )

    eval_ds = eval_ds.map(lambda x, y: (data_prep(x), y))

    print(
        f"Evaluation class distribution: {common.get_class_count(len(class_names), eval_ds )}"
    )

    return eval_ds


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages the `accuracy` metric sent by the clients in the `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""

    config = {
        "epochs": LOCAL_EPOCHS,  # Number of local epochs done by clients
        "batch_size": LOCAL_BATCH,  # Batch size to use by clients during fit()
    }
    return config


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    eval_df = pd.read_csv(VAL_DF_DIR)
    eval_ds = prepare_dataset_for_eval(eval_df)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(eval_ds)
        common.evaluate_model(model, eval_ds, os.path.join(RESULTS_DIR, "Evaluate"))
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    args = parser.parse_args()

    model = SqueezeNet(include_top=False, input_shape=(224, 224, 3))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.002),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # TODO: this is a workaround to set global variables
    # Set global variables
    global LOCAL_BATCH, LOCAL_EPOCHS
    LOCAL_EPOCHS = args.local_epochs
    LOCAL_BATCH = args.local_batch

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        min_evaluate_clients=args.min_num_clients,
        min_available_clients=args.min_num_clients,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
