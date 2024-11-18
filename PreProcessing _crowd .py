#!/usr/bin/env python
# coding: utf-8

# ## Processing 

# In[6]:


import os
import cv2
import scipy.io as sio
import numpy as np
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras import layers, models


# In[3]:


# Define paths
image_path = r"C:\Users\aryan\Desktop\ShanghaiTech\part_A\train_data\images"
label_path = r"C:\Users\aryan\Desktop\ShanghaiTech\part_A\train_data\ground-truth"


# In[4]:


# Function to load images and annotations
def load_data(image_path, label_path, sample_index=0):
    # List image files
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.mat')])
    
    # Load a sample image and corresponding label
    sample_image_file = image_files[26]
    sample_label_file = label_files[26]
    
    # Load image
    image = cv2.imread(os.path.join(image_path, sample_image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load label (ground-truth points)
    label_data = sio.loadmat(os.path.join(label_path, sample_label_file))
    points = label_data["image_info"][0,0][0,0][0]  # Adjust according to .mat structure

    return image, points


# In[5]:


# Load a sample image and display it with ground-truth points
image, points = load_data(image_path, label_path, sample_index=0)

plt.imshow(image)
plt.scatter(points[:, 0], points[:, 1], s=10, color='red')  # Plot points on image
plt.title("Sample Image with Ground-Truth Points")
plt.axis('off')
plt.show()


# In[6]:


def create_density_map(image_shape, points):
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    
    # Apply a Gaussian filter on each point
    for point in points:
        x, y = min(int(point[0]), image_shape[1] - 1), min(int(point[1]), image_shape[0] - 1)
        density_map[y, x] = 1

    density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
    return density_map


# In[7]:


# Create and display density map for the sample image
density_map = create_density_map(image.shape, points)

plt.imshow(density_map, cmap='hot')
plt.axis('off')
plt.title("Density Map for Sample Image")
plt.show()


# In[7]:


# Define paths
image_dir = r"C:\Users\aryan\Desktop\ShanghaiTech\part_A\train_data\images"
label_dir = r"C:\Users\aryan\Desktop\ShanghaiTech\part_A\train_data\ground-truth"


# In[10]:


def preprocess_image_and_density_map(img_path, label_path, target_size=(512, 512)):
    # Load and resize image
    image = Image.open(img_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]

    # Load .mat file and create density map
    ann_points = loadmat(label_path)['image_info'][0][0][0][0][0]
    density_map = np.zeros(target_size, dtype=np.float32)

    # Generate density map from annotated points
    for point in ann_points:
        x, y = min(int(point[0] * target_size[1] / image.shape[1]), target_size[1] - 1), \
               min(int(point[1] * target_size[0] / image.shape[0]), target_size[0] - 1)
        density_map[y, x] = 1
    
    density_map = gaussian_filter(density_map, sigma=15)  # Apply Gaussian filter

    return image, density_map

# Load and preprocess a sample image and density map to check
sample_images = os.listdir(image_dir)[:5]  # Load a few sample images
for img_name in sample_images:
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, f"GT_{img_name.split('.')[0]}.mat")
    img, density_map = preprocess_image_and_density_map(img_path, label_path)

    # Display the image and density map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title("Density Map")
    plt.show()

