import tensorflow as tf
import os

# Specify the path to the flower_photos directory
dataset_dir = "flower_photos"

# Get the list of subdirectories within the flower_photos directory
subdirectories = os.listdir(dataset_dir)

# Create a list to store the file paths for each image
image_paths = []

# Create a list to store the labels for each image
labels = []

# Loop through each subdirectory
for subdir in subdirectories:
    # Create the full path to the subdirectory
    subdir_path = os.path.join(dataset_dir, subdir)
    
    # Get a list of all the image files in the subdirectory
    image_files = os.listdir(subdir_path)
    
    # Loop through each image file in the subdirectory
    for image_file in image_files:
        # Create the full path to the image file
        image_path = os.path.join(subdir_path, image_file)
        
        # Add the image path to the list of image paths
        image_paths.append(image_path)
        
        # Add the label index to the list of labels
        labels.append(subdirectories.index(subdir))

# Convert the image paths and labels to tensors
image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)

# Create a dataset from the image paths and labels
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
