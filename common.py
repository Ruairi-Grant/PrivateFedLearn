'''Module to define common functions for the data processing and model training'''
import os
import tensorflow as tf


def configure_for_performance(ds, batch_size):
    '''Function to configure the dataset for performance'''
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds  #


def get_label(file_path, classes):
    '''Function to get the label of the image from the file path'''
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == classes
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img, im_size):
    '''Function to decode the image from the file path and resize it to the desired size'''
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, im_size)


def process_path(file_path, classes, im_size):
    '''General function to process the image'''
    label = get_label(file_path, classes)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, im_size)
    return img, label
