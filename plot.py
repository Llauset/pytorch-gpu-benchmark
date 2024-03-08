import pandas as pd
import os
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torchvision.models as models

MODEL_LIST = {
    models.resnet: ['resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2'],
    models.densenet: ['densenet121', 'densenet201'],
}

folder_name = 'result/'
csv_list = glob.glob(folder_name + '/*.csv')
columes = []

for key, values in MODEL_LIST.items():
    for i in values:
        columes.append((key, i))

train_data = {}  # Dictionary to store combined train data
inference_data = {}  # Dictionary to store combined inference data

# Define custom color palette
# colors = list(mcolors.TABLEAU_COLORS.values())
colors = plt.cm.tab20.colors
for csv in csv_list:
    df = pd.read_csv(csv)
    if "train" in csv:
        columes_filtered = [column for column in columes if "Weights" not in column[1]]
        df_mean = df.groupby(level=0, axis=1).mean().mean()

        train_data[csv] = df_mean
    elif "inference" in csv:
        columes_filtered = [column for column in columes if "Weights" not in column[1]]
        df_mean = df.groupby(level=0, axis=1).mean().mean()

        inference_data[csv] = df_mean

# Plotting train data
plt.figure(figsize=(10, 6))
for i, csv in enumerate(sorted(train_data.keys())):
    plt.scatter(train_data[csv].index, train_data[csv].values, label=csv.split('/')[1].split('_benchmark')[0], color=colors[i % len(colors)])

plt.title("Train Data")
plt.xlabel('Models')
plt.ylabel('Time (ms)')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plotting inference data
plt.figure(figsize=(10, 6))
for i, csv in enumerate(sorted(inference_data.keys())):
    plt.scatter(inference_data[csv].index, inference_data[csv].values, label=csv.split('/')[1].split('_benchmark')[0], color=colors[i % len(colors)])

plt.title("Inference Data")
plt.xlabel('Models')
plt.ylabel('Time (ms)')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

