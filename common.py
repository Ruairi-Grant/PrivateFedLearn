"""Module to define common functions for the data processing and model training"""

# General imports
import os
import numpy as np
import tensorflow as tf

# SKlearn model evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns


def configure_for_performance(ds, batch_size):
    """Function to configure the dataset for performance"""
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds  #


def get_label(file_path, classes):
    """Function to get the label of the image from the file path"""
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == classes
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img, im_size):
    """Function to decode the image from the file path and resize it to the desired size"""
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, im_size)


def process_path(file_path, classes, im_size):
    """General function to process the image"""
    label = get_label(file_path, classes)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, im_size)
    return img, label


def get_class_count(num_classes, dataset):
    """Function to get the class count of the dataset"""
    count = np.zeros(num_classes, dtype=np.int32)
    for _, labels in dataset:
        y, _, c = tf.unique_with_counts(labels)
        count[y.numpy()] += c.numpy()
    return count


def evaluate_model(model, dataset, dir_path):
    """Function to evaluate the model and save the confusion matrix and classification report"""
    _, eval_acc = model.evaluate(dataset, verbose=2)
    print("\nTrain accuracy:", eval_acc)
    # Clear the current matplotlib figure
    plt.clf()
    # Get the true labels and predicted labels
    true_labels = []
    predicted_labels = []

    for images, labels in dataset:
        predictions = model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))
        true_labels.extend(labels.numpy())

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(dir_path + "/confusion_matrix.png")

    # Create the classification report
    report = classification_report(true_labels, predicted_labels)

    with open(dir_path + "/classification_report.txt", "w") as file:
        file.write(report)
