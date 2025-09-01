import os
import sqlite3
import numpy as np
import faiss
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import argparse
import json

# Initialize database
DB_PATH = "cluster.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY, 
                  path TEXT UNIQUE, 
                  cluster_id INTEGER, 
                  processed INTEGER DEFAULT 0, 
                  issue TEXT)''')
    conn.commit()
    return conn

def add_images_to_db(conn, image_paths):
    c = conn.cursor()
    for path in image_paths:
        c.execute('INSERT OR IGNORE INTO images (path) VALUES (?)', (str(path),))
    conn.commit()

def load_embeddings(embedding_dir):
    index_path = os.path.join(embedding_dir, "embeddings.index")
    metadata_path = os.path.join(embedding_dir, "metadata.json")

    index = faiss.read_index(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return index, metadata['image_paths']

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

    # Stack using median
    stacked = np.median([img.astype(np.float32) for img in images], axis=0).astype(np.uint8)
    cv2.imwrite(str(output_path), stacked)
    return True

def process_cluster(conn, cluster_images):
    c = conn.cursor()
    image_paths = [Path(row[0]) for row in cluster_images]
    output_path = image_paths[0].parent / f"cluster_{image_paths[0].stem}.png"

    if not stack_images(image_paths, output_path):
        return False

    # Check sharpness improvement
    base_sharpness = calculate_sharpness(image_paths[0])
    stacked_sharpness = calculate_sharpness(output_path)

    if stacked_sharpness <= base_sharpness * 0.9:
        print(f"Cluster {image_paths[0].stem} stacking didn't improve sharpness")
        return False

    # Mark as processed
    for path, _ in cluster_images:
        c.execute('UPDATE images SET processed=1 WHERE path=?', (str(path),))
    conn.commit()
    return True

def find_best_cluster(embeddings, unprocessed_ids):
    # Find optimal DBSCAN parameters
    eps = 0.5  # Will need tuning
    min_samples = 2
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = clustering.labels_

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Filter out noise (-1)
    valid_labels = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]

    if len(valid_labels) == 0:
        return None

    # Find the label with the maximum count
    best_label = valid_labels[np.argmax(valid_counts)]

    # Return indices of the best cluster
    cluster_mask = labels == best_label
    return np.array(unprocessed_ids)[cluster_mask]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster images using embeddings')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Folder with embeddings')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save clusters')

    args = parser.parse_args()

    # Initialize
    os.makedirs(args.output_dir, exist_ok=True)
    conn = init_db()

    # Load embeddings and image paths
    index, image_paths = load_embeddings(args.embedding_dir)
    add_images_to_db(conn, image_paths)

    while True:
        # Get unprocessed images
        c = conn.cursor()
        c.execute('SELECT path FROM images WHERE processed=0')
        unprocessed_paths = [Path(row[0]) for row in c.fetchall()]
        if not unprocessed_paths:
            break

        # Get their embeddings
        unprocessed_indices = [image_paths.index(str(p)) for p in unprocessed_paths]
        unprocessed_embeddings = index.reconstruct_n(0, len(image_paths))[unprocessed_indices]

        # Find best cluster
        cluster_indices = find_best_cluster(unprocessed_embeddings, unprocessed_indices)
        if cluster_indices is None:
            break

        # Get image paths for this cluster
        cluster_images = [(image_paths[i], i) for i in cluster_indices]

        # Process and save
        if process_cluster(conn, cluster_images):
            print(f"Processed cluster with {len(cluster_images)} images")
        else:
            # Mark as problematic
            for path, _ in cluster_images:
                c.execute('UPDATE images SET issue=?, processed=1 WHERE path=?', 
                         ("stacking_failed", str(path)))
            conn.commit()

    print("Clustering completed")


