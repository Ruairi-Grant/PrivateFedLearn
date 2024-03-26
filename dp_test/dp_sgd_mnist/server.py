import argparse
import os
from typing import List, Tuple

import tensorflow as tf

import flwr as fl
from flwr.common import Metrics

import common

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages teh `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main(args) -> None:
    model = common.create_cnn_model()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile("sgd", loss=loss, metrics=["accuracy"])
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_available_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(
        server_address="192.168.0.10:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=10, type=int)
    parser.add_argument("--fraction-fit", default=1.0, type=float)
    args = parser.parse_args()
    main(args)
