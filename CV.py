import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set the absolute path to the main directory
dataset_path = Path("D:/Colleage/7th-semester/CV/project/Data/Product Classification")

# Function to extract SIFT features from an image
def extract_sift_features(image_path):
    img = cv2.imread(str(image_path))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return sift, descriptors

# Collect SIFT features and labels from all images in the dataset
sift_features = []
labels = []

for class_label in range(1, 21):
    class_folder = str(class_label)
    class_path = dataset_path / class_folder / 'Train'

    for image_path in class_path.glob("*"):
        sift, sift_descriptor = extract_sift_features(image_path)
        sift_features.extend(sift_descriptor)
        labels.append(class_folder)

# Convert the list of SIFT features to a NumPy array
sift_features = np.array(sift_features)

# Perform k-means clustering to create the visual vocabulary
kmeans = KMeans(n_clusters=200, n_init=10)
kmeans.fit(sift_features)
visual_words = kmeans.cluster_centers_

# Function to represent an image using the BoW model
def image_to_bow(image_path, kmeans, sift):
    img = cv2.imread(str(image_path))
    keypoints, descriptors = sift.detectAndCompute(img, None)
    bow_descriptor = np.zeros(len(visual_words))

    for descriptor in descriptors:
        idx = np.argmin(np.linalg.norm(visual_words - descriptor, axis=1))
        bow_descriptor[idx] += 1

    return bow_descriptor

# Collect BoW representations for all images in the dataset
bow_features = []

for class_label in range(1, 21):
    class_folder = str(class_label)
    class_path = dataset_path / class_folder / 'Train'

    for image_path in class_path.glob("*"):
        sift, _ = extract_sift_features(image_path)
        bow_descriptor = image_to_bow(image_path, kmeans, sift)
        bow_features.append(bow_descriptor)

# Convert the list of BoW features to a NumPy array
bow_features = np.array(bow_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bow_features, labels, test_size=0.2, random_state=42)

# Create a Random Forest classifier and train it
rf = RandomForestClassifier(n_estimators=250)
rf.fit(X_train, y_train)

# Evaluate the classifier
accuracy = rf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")