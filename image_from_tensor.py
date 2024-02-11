import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_DIR = "C:\\git_repos\\Thesis\\data\\diabetic_retinopathy"
DF_DIR = DATA_DIR + "\\train_set.csv"

# Defing Hyperparamaters
BATCH_SIZE = 50
SEED = 42
# Load dataframe
df = pd.read_csv(DF_DIR)

# Split dataframe into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)


def load_image(file_name):
    raw = tf.io.read_file(file_name)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    return tensor


def create_dataset(file_names, labels):
    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
    dataset = dataset.map(lambda file_name, label: (load_image(file_name), label))
    return dataset

df["file_name"] = DATA_DIR + "\\images\\" + df["id_code"] + ".png"
file_names = df["file_name"].to_numpy()
labels = df["diagnosis"].to_numpy()
dataset = create_dataset(file_names, labels)
