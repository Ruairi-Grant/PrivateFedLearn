import pandas as pd
import shutil
#TODO: test this
# Assuming your dataframe is named df and the category column is named 'category'
df= pd.read_csv('C:\\git_repos\\Thesis\\Datasets\\aptos2019-blindness-detection\\train.csv')

import pandas as pd
import numpy as np
from pathlib import Path

# Assuming your dataframe is named df and the category column is named 'category'


import pandas as pd
import numpy as np

# Assuming your dataframe is named df and the category column is named 'diagnosis'
DATA_DIR = "data/diabetic_retinopathy/"

# Step 1: Group by the diagnosis column
grouped = df.groupby('diagnosis')

# Step 2: Sort each group by the diagnosis column
sorted_groups = {k: v.sort_values('diagnosis') for k, v in grouped}

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
        subgroup = v.iloc[start:start+subgroup_size]
        subgroups[k].append(subgroup)
        start += subgroup_size

# Step 4: Combine subgroups from different diagnoses into the final groups
final_groups = []
for i in range(num_subgroups):
    final_group = pd.concat([subgroups[k][i] for k in sorted_groups.keys()])
    final_groups.append(final_group)

# Now final_groups contains your four groups with slightly unequal sizes
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

split_data_into_classes(final_groups[0], 'Client1_train')
split_data_into_classes(final_groups[1], 'Client2_train')
split_data_into_classes(final_groups[2], 'Client3_train')
split_data_into_classes(final_groups[3], 'Client4_train')
