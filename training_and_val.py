import pandas as pd

DATA_DIR = "C:\\git_repos\\Thesis\\Datasets\\aptos2019-blindness-detection\\train.csv"

# Load the data from the CSV file
data_df = pd.read_csv(DATA_DIR)

# Group by the diagnosis column
data_df = data_df.groupby("diagnosis")

# Sort each group by the diagnosis column
sorted_groups = {k: v.sort_values("diagnosis") for k, v in data_df}

# Define the split percentage
split_percentage = 0.9  # 90/10 split

# Split each group into two groups with the specified split percentage across each diagnosis
num_subgroups = 2
subgroups = {k: [] for k in sorted_groups.keys()}
for k, v in sorted_groups.items():
    group_size = len(v)
    # Calculate the size of the first subgroup
    subgroup_size_1 = int(group_size * split_percentage)
    # Add data to the first subgroup
    subgroups[k].append(v.iloc[:subgroup_size_1])
    # Add the remaining data to the second subgroup
    subgroups[k].append(v.iloc[subgroup_size_1:])

# Combine subgroups from different diagnoses into the final groups
final_groups = []
for i in range(num_subgroups):
    final_group = pd.concat([subgroups[k][i] for k in sorted_groups.keys()])
    final_groups.append(final_group)

# Print the size of each subgroup for both final groups
for i, final_group in enumerate(final_groups):
    print(f"Size of Final Group {i + 1}: {len(final_group)}")
    for j, subgroup in enumerate(subgroups.values()):
        print(f"  Size of Subgroup {j + 1}: {len(subgroup[i])}")

# Save the final groups to CSV files
final_groups[0].to_csv(
    "C:\\git_repos\\Thesis\\Datasets\\aptos2019-blindness-detection\\split_train.csv",
    index=False,
)
final_groups[1].to_csv(
    "C:\\git_repos\\Thesis\\Datasets\\aptos2019-blindness-detection\\split_val.csv",
    index=False,
)
