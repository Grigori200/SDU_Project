import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from PIL import Image

# Author: Adrian

# Prepare dataframe of images in the provided directory.
filenames_normal = [f'data/{i}_normal.jpg' for i in range(1, 1101)]
filenames_positive = [f'data/{i}_pneumonia.jpg' for i in range(1, 1101)]
labels = list(np.zeros(1100)) + list(np.ones(1100))
filenames = filenames_normal + filenames_positive
indexes = np.arange(2200)

# Split the data into training, validation and test sets.
train_idxs, test_idxs = train_test_split(indexes, test_size=0.2, random_state=42)
train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.15, random_state=42)
splits = np.empty(2200, dtype='object')
splits[train_idxs] = "train"
splits[val_idxs] = "val"
splits[test_idxs] = "test"

filenames = np.array(filenames)
labels = np.array(labels)
df_filenames = list(filenames[train_idxs]) + list(filenames[val_idxs]) + list(filenames[test_idxs])
df_splits = list(splits[train_idxs]) + list(splits[val_idxs]) + list(splits[test_idxs])
df_labels = list(labels[train_idxs]) + list(labels[val_idxs]) + list(labels[test_idxs])
df = pd.DataFrame({
    'filename': df_filenames,
    'splits': df_splits,
    'labels': df_labels
})
df.to_csv('data/data.csv')

# Split the data into proper directiories.
train_data = []
val_data = []
test_data = []

df = pd.read_csv('data/data.csv')
train_filenames = df[df['splits'] == 'train']['filename'].tolist()
val_filenames = df[df['splits'] == 'val']['filename'].tolist()
test_filenames = df[df['splits'] == 'test']['filename'].tolist()

train_data = [np.array(Image.open(path)) for path in train_filenames]
val_data = [np.array(Image.open(path)) for path in val_filenames]
test_data = [np.array(Image.open(path)) for path in test_filenames]

print(f'Train: {train_data.__len__()} Val: {val_data.__len__()} Test: {test_data.__len__()}')

all_data = train_data + val_data + test_data

dir_df_filenames = []
for filename, split, label, image in zip(df_filenames, df_splits, df_labels, all_data):
  filename = filename.split('/')[-1]
  if split == 'train':
    if label == 0:
      filename = f'split_data/training/normal/{filename}'
      img = Image.fromarray(image)
      img.save(filename)
    if label == 1:
      filename = f'split_data/training/pneumonia/{filename}'
      img = Image.fromarray(image)
      img.save(filename)
  elif split == 'val':
    if label == 0:
      filename = f'split_data/validation/normal/{filename}'
      img = Image.fromarray(image)
      img.save(filename)
    if label == 1:
      filename = f'split_data/validation/pneumonia/{filename}'
      img = Image.fromarray(image)
      img.save(filename)
  elif split == 'test':
    if label == 0:
      filename = f'split_data/testing/normal/{filename}'
      img = Image.fromarray(image)
      img.save(filename)
    if label == 1:
      filename = f'split_data/testing/pneumonia/{filename}'
      img = Image.fromarray(image)
      img.save(filename)
  dir_df_filenames.append(filename)

# Save the modified dataframe with paths to moved images.
dir_df = pd.DataFrame({
    'filename': dir_df_filenames,
    'splits': df_splits,
    'labels': df_labels
})
dir_df.to_csv('data_split_to_dirs.csv')
