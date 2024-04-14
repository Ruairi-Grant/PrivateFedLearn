import argparse
import os
from typing import List, Dict, Optional, Tuple, Union

import numpy as np

# SKlearn model evaluation
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import tensorflow as tf

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Scalar,
    Metrics,
)
from flwr.server.client_proxy import ClientProxy

import common

RESULTS_DIR = os.path.join("Results", "Federated_mnist")
num_rounds = 1  # global that will be updated in main, this is probably bad practice
loss_history = []
accuracy_history = []

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def evaluate_model(eval_model, X, y, dir_path):
    """Function to evaluate the model and save the confusion matrix and classification report"""
    _, eval_acc = eval_model.evaluate(X, y, verbose=1)
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

    with open(
        os.path.join(dir_path, "classification_report"), "w", encoding="utf-8"
    ) as file:
        file.write(report)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate the metrics from the clients using a weighted average."""
    # Multiply accuracy of each client by number of examples used
    train_acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    train_losses = [num_examples * m["loss"] for num_examples, m in metrics]
    val_acc = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    results = {
        "loss": sum(train_losses) / sum(examples),
        "accuracy": sum(train_acc) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_acc) / sum(examples),
    }

    # Aggregate and return custom metric (weighted average)
    return results



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

        if server_round == num_rounds:
            evaluate_model(model, x_test, y_test, os.path.join(RESULTS_DIR, "Test"))

        # inneficeint to calculate this twice but not a big deal
        loss, accuracy = model.evaluate(x_test, y_test)

        return loss, {"accuracy": accuracy, "loss": loss}

    return evaluate


def main(args) -> None:
    model = common.create_cnn_model()
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True
    )  # Loss doesnt matter heve because its only for inference
    model.compile("sgd", loss=loss, metrics=["accuracy"])

    global num_rounds

    num_rounds = args.num_rounds
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_available_clients=args.num_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(model, args.num_clients),
        # evaluate_metrics_aggregation_fn=weighted_average,
    )
    history = fl.server.start_server(
        server_address=args.server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

    # check if the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Save the federated loss and accuracy
    agg_loss_epoch = [item[0] for item in history.metrics_distributed_fit["loss"]]
    agg_loss = [item[1] for item in history.metrics_distributed_fit["loss"]]
    agg_val_loss = [item[1] for item in history.metrics_distributed_fit["val_loss"]]

    agg_acc_epoch = [item[0] for item in history.metrics_distributed_fit["accuracy"]]
    agg_acc = [item[1] for item in history.metrics_distributed_fit["accuracy"]]
    agg_val_acc = [item[1] for item in history.metrics_distributed_fit["val_accuracy"]]

    # save the centralized loss and accuracy
    cen_acc_epoch = [item[0] for item in history.metrics_centralized["accuracy"]]
    cen_acc = [item[1] for item in history.metrics_centralized["accuracy"]]
    cen_loss = [item[1] for item in history.metrics_centralized["loss"]]

    # plot the federated loss
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    ax1.plot(agg_loss_epoch, agg_loss, label="Training Loss")
    ax1.plot(agg_loss_epoch, agg_val_loss, label="Validation Loss")
    ax1.legend(loc="lower right")

    fig1.savefig(os.path.join(RESULTS_DIR, "FederatedLoss"))

    # plot the federated accuracy
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    ax2.plot(agg_acc_epoch, agg_acc, label="Training Accuracy")
    ax2.plot(agg_acc_epoch, agg_val_acc, label="Validation Accuracy")
    ax2.legend(loc="lower right")

    fig2.savefig(os.path.join(RESULTS_DIR, "FederatedAccuracy"))

    # PLot the centralized accuracy and save it
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    ax3.plot(cen_acc_epoch, cen_acc, label="Validation Accuracy")
    ax3.legend(loc="lower right")

    fig3.savefig(os.path.join(RESULTS_DIR, "Centralized_TrainingValidationAccuracy"))

    # PLot the centralized loss and save it
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    ax4.plot(cen_acc_epoch, cen_loss, label="Validation Loss")
    ax4.legend(loc="lower right")
    fig4.savefig(os.path.join(RESULTS_DIR, "Centralized_TrainingValidationLoss"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=2, type=int)
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--fraction-fit", default=1.0, type=float)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()
    main(args)
