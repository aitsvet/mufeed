import os
import numpy as np
import faiss
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import csv
import sys
import json

embedding_dir = sys.argv[1]
index_path = os.path.join(embedding_dir, "embeddings.index")
metadata_path = os.path.join(embedding_dir, "metadata.json")
index = faiss.read_index(index_path)
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
image_paths = metadata['image_paths']

output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "cluster.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'cluster_id', 'processed', 'issue'])

unprocessed_paths = image_paths.copy()

def calculate_sharpness(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def stack_images(image_paths, output_path):
    images = [cv2.imread(str(p)) for p in image_paths if os.path.exists(p)]
    if not images: return None
    if len(images) == 1:
        cv2.imwrite(str(output_path), images[0])
        return True
    stacked = np.median([img.astype(np.float32) for img in images], axis=0).astype(np.uint8)
    cv2.imwrite(str(output_path), stacked)
    return True

while unprocessed_paths:
    unprocessed_indices = [image_paths.index(str(p)) for p in unprocessed_paths]
    unprocessed_embeddings = index.reconstruct_n(0, len(image_paths))[unprocessed_indices]

    # Find optimal eps for current unprocessed embeddings
    eps = 0.8  # Start with a high eps to get small clusters
    best_cluster_size = 0
    max_attempts = 10
    eps_step = 0.1
    
    for _ in range(max_attempts):
        clustering = DBSCAN(eps=eps, min_samples=2).fit(unprocessed_embeddings)
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[unique_labels != -1]
        valid_counts = counts[unique_labels != -1]
        if eps < eps_step:
            break
        if len(valid_counts) == 0:
            eps -= eps_step
            continue
        max_count = valid_counts[np.argmax(valid_counts)]
        if max_count <= 8 and max_count >= 4:
            break
        if max_count < 4:
            eps -= eps_step
        else:
            eps += eps_step
    
    # Get the best cluster size from the final eps
    clustering = DBSCAN(eps=eps, min_samples=2).fit(unprocessed_embeddings)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]
    
    if len(valid_counts) == 0:
        break
    
    max_count = valid_counts[np.argmax(valid_counts)]
    print(f"Using eps={eps} which gives cluster size={max_count}")

    # Find the label with the maximum count
    best_label = valid_labels[np.argmax(valid_counts)]

    # Get indices of the best cluster
    cluster_mask = labels == best_label
    cluster_indices = np.array(unprocessed_indices)[cluster_mask]
    
    # Get image paths for this cluster
    cluster_images = [image_paths[i] for i in cluster_indices]

    # Process and save
    output_path = Path(cluster_images[0]).parent / f"cluster_{Path(cluster_images[0]).stem}.png"
    
    if not stack_images(cluster_images, output_path):
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for path in cluster_images:
                writer.writerow([path, None, 1, "stacking_failed"])
        break

    # Check sharpness improvement
    base_sharpness = calculate_sharpness(cluster_images[0])
    stacked_sharpness = calculate_sharpness(output_path)

    if stacked_sharpness <= base_sharpness * 0.9:
        print(f"Cluster {Path(cluster_images[0]).stem} stacking didn't improve sharpness")
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for path in cluster_images:
                writer.writerow([path, None, 1, "sharpness_not_improved"])
        break

    # Mark as processed
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for path in cluster_images:
            writer.writerow([path, cluster_images[0], 1, None])
    
    # Update unprocessed_paths
    unprocessed_paths = [p for p in unprocessed_paths if p not in cluster_images]

print("Clustering completed")


