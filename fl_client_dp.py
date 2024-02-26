from pathlib import Path
import argparse
import warnings


import flwr as fl
from flwr.common import NDArrays
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
    VectorizedDPKerasAdamOptimizer,
)
from keras import layers as tfkl


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

parser.add_argument("--l2-norm-clip", default=1.0, type=float, help="Clipping norm")

parser.add_argument(
    "--noise-multiplier",
    default=1.1,
    type=float,
    help="Ratio of the standard deviation to the clipping norm",
)


warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 4
DATA_DIR = "data/diabetic_retinopathy/"
SEED = 42
BATCH_SIZE = 50
LOCAL_EPOCHS = 10

# global for tracking privacy
PRIVACY_LOSS = 0


def prepare_dataset():
    """Download and partitions the CIFAR-10/MNIST dataset."""
    # TODO: I need to load the data here and then partitian it
    # TODO: This is also where data augmentation needs to happen
    partitions = []

    data_prep = tf.keras.Sequential(
        [
            # tfkl.Rescaling(1./255),
            tfkl.CenterCrop(224, 224)
        ]
    )

    data_augmentation = tf.keras.Sequential([data_prep, tfkl.RandomFlip("horizontal")])

    for cid in range(NUM_CLIENTS):

        dir_name = "Client" + str(cid + 1) + "_train"
        train_ds = tf.keras.utils.image_dataset_from_directory(
            Path(DATA_DIR).joinpath(dir_name),
            validation_split=0.1,
            subset="training",
            color_mode="rgb",
            seed=SEED,
            image_size=(
                265,
                265,
            ),  # This resizes the image, using bilinear transformation
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            Path(DATA_DIR).joinpath(dir_name),
            validation_split=0.1,
            subset="validation",
            color_mode="rgb",
            seed=SEED,
            image_size=(
                265,
                265,
            ),  # This resizes the image, using bilinear transformation
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        # prepare the datasets
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
        val_ds = val_ds.map(lambda x, y: (data_prep(x), y))

        partitions.append((train_ds, val_ds))

    return partitions

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
    sampling_probability = batch_size / num_train_examples
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )
    # Delta is set to approximate 1 / (number of training points).
    return get_privacy_spent(orders, rdp, target_delta=1 / num_train_examples)[0]

class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that uses SqueeseNet."""

    def __init__(self, trainset, valset, noise_multiplier):
        self.x_train_ds = trainset
        self.x_val_ds = valset
        # Instantiate model
        self.model = SqueezeNet(include_top=False, input_shape=(224, 224, 3))
        

        # Differential Privacy stuff
        self.noise_multiplier = noise_multiplier

        optimizer = VectorizedDPKerasAdamOptimizer(
            noise_multiplier,
            num_microbatches=None,
            unconnected_gradients_to_zero=False
        )

        self.model.compile(
            optimizer=optimizer,
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

        global PRIVACY_LOSS

        privacy_spent = compute_epsilon(
            LOCAL_EPOCHS,
            int(self.x_val_ds.cardinality().numpy()),
            BATCH_SIZE,
            self.noise_multiplier,
        )
        PRIVACY_LOSS += privacy_spent
            
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
    partitions = prepare_dataset()
    trainset, valset = partitions[args.cid]

    # Start Flower client setting its associated data partition
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainset, valset=valset, noise_multiplier=args.noise_multiplier
        ),
    )

    print("Privacy Loss: ", PRIVACY_LOSS)


if __name__ == "__main__":
    main()
