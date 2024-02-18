import cv2
import os
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import shutil

# Ustaw random seed
np.random.seed(374)

def extract_features(images):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    features_list = []
    for img in images:
        img = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array)
        features = features.flatten()

        features_list.append(features)

    return np.array(features_list)

def create_visual_dictionary(features, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    return kmeans.cluster_centers_

def vector_quantization(features, visual_dictionary):
    # Calculate distances to visual dictionary centroids
    distances = np.linalg.norm(features[:, np.newaxis, :] - visual_dictionary, axis=2)

    # Assign each feature to the nearest visual word
    nearest_visual_words = np.argmin(distances, axis=1)

    # Build histogram of visual words
    histogram, _ = np.histogram(nearest_visual_words, bins=np.arange(len(visual_dictionary)+1))

    return histogram

def detect_outliers(features, cluster_label, method='threshold', contamination=0.01):
    if method == 'isolation_forest':
        clf = IsolationForest(contamination=contamination, random_state=94)
        outliers = clf.fit_predict(features)
        return outliers == -1
    elif method == 'threshold':
        # Set a threshold of 60 for Cluster_7
        if cluster_label == 7:
            threshold = 42
        elif cluster_label ==0:
            threshold = 54.5
        else:
            # For other clusters, use the 90th percentile
            threshold = np.percentile(np.linalg.norm(features, axis=1), 100)
        return np.linalg.norm(features, axis=1) > threshold
    else:
        raise ValueError(f"Nieznana metoda detekcji outlierów: {method}")


def main(folder_path, output_folder):
    images = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            images.append(img)
            filenames.append(filename)

    # Ekstrahuj cechy przy użyciu modelu ResNet-50
    features = extract_features(images)

    # Stwórz słownik słów wzorcowych (słów kluczowych)
    visual_dictionary = create_visual_dictionary(features, k=8)

    # Zastosuj k-means do cech obrazów
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(features)

    # Przypisz etykiety klastrów do obrazów
    image_labels = kmeans.predict(features)

    # Pogrupuj obrazy według klastrów
    grouped_images = {}
    for i, label in enumerate(image_labels):
        if label not in grouped_images:
            grouped_images[label] = [(filenames[i], images[i])]
        else:
            grouped_images[label].append((filenames[i], images[i]))

    # Utwórz foldery dla klastrów
    for label in range(8):
        cluster_folder = os.path.join(output_folder, f"Cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)

        # Przenieś obrazy do odpowiednich folderów i zastosuj Visual Bag of Words
    for label, images_in_cluster in grouped_images.items():
        cluster_folder = os.path.join(output_folder, f"Cluster_{label}")

        # Extract features for images in the cluster
        cluster_features = extract_features([img for _, img in images_in_cluster])

        # Perform outlier detection with the specific threshold for Cluster_7
        outliers = detect_outliers(cluster_features, cluster_label=label, method='isolation_forest' if label == 3  else 'threshold')

        # Copy images to cluster folder
        for (filename, img), outlier in zip(images_in_cluster, outliers):
            destination_folder = os.path.join(cluster_folder, "Outliers") if outlier else cluster_folder
            os.makedirs(destination_folder, exist_ok=True)

            destination_path = os.path.join(destination_folder, filename)
            shutil.copy(os.path.join(folder_path, filename), destination_path)

if __name__ == "__main__":
    folder_path = "./Final_Images_dataset"
    output_folder = "./klastry"
    main(folder_path, output_folder)