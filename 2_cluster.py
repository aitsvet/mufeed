import os
import sys
import json
import shutil
import logging
import numpy as np
from pathlib import Path
import cv2
import faiss
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_sharpness(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def cluster_images(embedding_dir, output_dir):
    index_path = os.path.join(embedding_dir, "embeddings.index")
    metadata_path = os.path.join(embedding_dir, "metadata.json")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    embeddings = index.reconstruct_n(0, index.ntotal)
    image_paths = metadata['image_paths']
    embeddings = normalize(embeddings, axis=1)
    
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
    labels = clustering.labels_
    
    clusters = {}
    for path, label in zip(image_paths, labels):
        clusters.setdefault(label, []).append(path)
    
    for label, paths in clusters.items():
        if len(paths) > 50:
            logging.debug(f"Re-clustering large cluster {label} with {len(paths)} images")
            cluster_indices = [image_paths.index(p) for p in paths]
            cluster_embeddings = embeddings[cluster_indices]
            
            sub_clustering = DBSCAN(eps=0.2, min_samples=2).fit(cluster_embeddings)
            sub_labels = sub_clustering.labels_
            
            for sub_label in set(sub_labels):
                if sub_label == -1:
                    continue
                sub_paths = [paths[i] for i in np.where(sub_labels == sub_label)[0]]
                process_cluster(sub_paths, output_dir, f"{label}_{sub_label}")
        else:
            process_cluster(paths, output_dir, label)

def process_cluster(paths, output_dir, cluster_id):
    sorted_paths = sorted(paths)
    cluster_name = Path(sorted_paths[0]).stem
    cluster_dir = Path(output_dir) / cluster_name
    cluster_dir.mkdir(parents=True, exist_ok=True)
    
    sharpest_image = None
    max_sharpness = -1
    
    for path in sorted_paths:
        shutil.copy2(path, cluster_dir)
        sharpness = calculate_sharpness(path)
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            sharpest_image = path
    
    if sharpest_image:
        sharpest_filename = f"{cluster_name}_sharpest{Path(sharpest_image).suffix}"
        shutil.copy2(sharpest_image, Path(output_dir) / sharpest_filename)
    
    logging.debug(f"Cluster {cluster_id}: {len(paths)} images, sharpest: {Path(sharpest_image).name}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 2_cluster.py <embedding_dir> <output_dir>")
        sys.exit(1)
    
    embedding_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    cluster_images(embedding_dir, output_dir)
    logging.info("Clustering completed")