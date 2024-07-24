import os
import shutil
import random

# Set the seed for reproducibility

random.seed(42)

# Paths

base_path = 'mydata'
images_path = os.path.join(base_path, 'images')
labels_path = os.path.join(base_path, 'labels')

# Split Ratios

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Create directories for train, val, and test sets

for set_type in ['train', 'val', 'test']:
    for content_type in ['images', 'labels']:
        os.makedirs(os.path.join(base_path, set_type, content_type), exist_ok=True)

# Get all image filenames

all_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
random.shuffle(all_files)

# Calculate split indices

total_files = len(all_files)
train_end = int(train_ratio * total_files)
val_end = train_end + int(val_ratio * total_files)

# Split files

train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

# Function to copy files

def copy_files(files, set_type):
    for file in files: # Copy image
        shutil.copy(os.path.join(images_path, file), os.path.join(base_path, set_type, 'images')) # Copy corresponding label
        label_file = file.rsplit('.', 1)[0] + '.txt'
        shutil.copy(os.path.join(labels_path, label_file), os.path.join(base_path, set_type, 'labels'))

# Copy files to respective directories

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset successfully split into train, val, and test sets.")