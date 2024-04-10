"""Script to download the Diabetic Retinopathy data from google drive, 
and arrange it in a format for tensorflow image_dataset_from_directory()"""

import os

import urllib.request
import zipfile
from pathlib import Path
import shutil

import pandas as pd
import gdown


DATA_DIR = "data/diabetic_retinopathy/"
DATA_URL = "https://drive.google.com/uc?id=1G-4UhPKiQY3NxQtZiWuOkdocDTW6Bw0u"
TEST_CSV_URL = "https://drive.google.com/uc?id=1dmeYLURzEvx962th4lAxaVN3r6nlhTjS"
TRAIN_CSV_URL = "https://drive.google.com/uc?id=1SMb9CRHjB6UH2WnTZDFVSgpA6_nh75qN"


if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# download ZIP, unzip it, delete zip file
ZIP_DIR = DATA_DIR + "images.zip"
gdown.download(DATA_URL, ZIP_DIR, quiet=False)
print("Extracting...!")

with zipfile.ZipFile(ZIP_DIR, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)
print("Extracted!")
os.remove(ZIP_DIR)

# download train and test dataframes

TEST_CSV_DIR = DATA_DIR + "test_set.csv"
TRAIN_CSV_DIR = DATA_DIR + "train_set.csv"
urllib.request.urlretrieve(TRAIN_CSV_URL, TRAIN_CSV_DIR)
urllib.request.urlretrieve(TEST_CSV_URL, TEST_CSV_DIR)
df_train = pd.read_csv(TRAIN_CSV_DIR)
df_test = pd.read_csv(TEST_CSV_DIR)


def split_data_into_classes(data_df, data_type):
    """split all the images into the folder according to their label"""
    for diagnosis, group in data_df.groupby("diagnosis"):
        # Create the dir that images of a class will be stored in
        path_name = Path(DATA_DIR).joinpath(data_type, "class_" + str(diagnosis))
        if not path_name.is_dir():
            path_name.mkdir(parents=True)

        # Extract the path for each imagec
        image_class_names = list(group["id_code"])
        class_paths = [
            Path(DATA_DIR).joinpath("images", name + ".png")
            for name in image_class_names
        ]
        # Create a new path for each image in the corresponding dir
        new_paths = [path_name.joinpath(name + ".png") for name in image_class_names]
        print(f"Class: {diagnosis}, Count: {len(image_class_names)}")

        # copy the images across from the old location to the new one
        for src, dst in zip(class_paths, new_paths):
            shutil.copy(src, dst)


split_data_into_classes(df_train, "train")
split_data_into_classes(df_test, "test")
