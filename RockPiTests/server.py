import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np

# SKlearn model evaluation
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import tensorflow as tf

import flwr as fl
from flwr.common import Metrics

import common

RESULTS_DIR = os.path.join("Results", "Federated_mnist")
num_rounds = 1 # global that will be updated in main, this is probably bad practice

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def evaluate_model(eval_model, X, y, dir_path):
    """Function to evaluate the model and save the confusion matrix and classification report"""
    _, eval_acc = eval_model.evaluate(X , y, verbose=1)
    print("\nTrain accuracy:", eval_acc)
    # Clear the current matplotlib figure
    plt.clf()
    # Get the true labels and predicted labels
    true_labels = []
    predicted_labels = []

    predictions = eval_model.predict(X)
    predicted_labels.extend(np.argmax(predictions, axis=1))
    true_labels = np.argmax(y, axis=1)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(os.path.join(dir_path, "confusion_matrix.png"))

    # Create the classification report
    report = classification_report(true_labels, predicted_labels)

    # check if the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(dir_path, "classification_report"), "w") as file:
        file.write(report)




def get_evaluate_fn(model, num_clients):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    # Load creates and extra test partition, so we just take the last one
    _, (x_test, y_test) = common.load(num_clients)[-1]

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        
        model.set_weights(parameters)  # Update model with the latest parameters

        if server_round == num_rounds - 1:
             evaluate_model(model, x_test, y_test, os.path.join(RESULTS_DIR, "Train"))
        
        # inneficeint to calculate this twice but not a big deal
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


def main(args) -> None:
    model = common.create_cnn_model()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # Loss doesnt matter heve because its only for inference
    model.compile("sgd", loss=loss, metrics=["accuracy"])

    global num_rounds 
    num_rounds = args.num_rounds
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_available_clients=args.num_clients,
        evaluate_fn=get_evaluate_fn(model, args.num_clients),
    )
    fl.server.start_server(
        server_address=args.server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--fraction-fit", default=1.0, type=float)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()
    main(args)
