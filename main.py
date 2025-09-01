import os
import shutil
import numpy as np
import torch
import timm
from PIL import Image
import cv2
import faiss
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import json
import argparse
import tempfile
import subprocess
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

def check_dependencies():
    """Check if required external tools are available"""
    missing_deps = []
    
    # Check for standard dependencies
    for cmd, desc, version_flag in [
        ("convert", "ImageMagick (for PDF conversion)", "-version"),
        ("pdfunite", "Poppler-utils (for PDF merging)", "-v"),
        ("tesseract", "Tesseract OCR (for text recognition)", "--version"),
        ("ffmpeg", "FFmpeg (for video processing)", "-version")
    ]:
        try:
            # Special handling for convert command
            if cmd == "convert":
                subprocess.run([cmd, version_flag], 
                              capture_output=True, check=True)
            else:
                subprocess.run([cmd, version_flag], 
                              capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Only add ffmpeg to missing_deps if we might need it
            if cmd != "ffmpeg" or "--video" in sys.argv:
                missing_deps.append(desc)
    
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        install_cmd = "sudo apt-get install imagemagick poppler-utils tesseract-ocr"
        if "FFmpeg" in " ".join(missing_deps):
            install_cmd += " ffmpeg"
            
        print(f"\nInstall with: {install_cmd}")
        return False
    return True

def check_ffmpeg():
    """Check if ffmpeg is available (separate from main dependency check)"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_video_frames(video_path, output_folder, threshold=0.35, min_scene_len=1.0):
    """
    Extract frames from video using scene detection optimized for presentations
    
    Args:
        video_path: Path to video file
        output_folder: Folder to save extracted frames
        threshold: Scene change threshold (0-1, higher = less sensitive)
        min_scene_len: Minimum time (in seconds) for a scene to be considered valid
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n" + "="*50)
    print(f"VIDEO FRAME EXTRACTION")
    print("="*50)
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Scene change threshold: {threshold} (higher = less sensitive)")
    print(f"Note: For presentation videos, 0.3-0.4 is typically optimal")
    
    # Use ffmpeg with scene detection
    cmd = [
        "ffmpeg",
        "-i", video_path,
        # Detect scene changes with specified threshold
        # gt(scene,{threshold}) selects frames after a scene change
        "-vf", f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
        "-vsync", "vfr",  # Variable frame rate output
        "-frame_pts", "1",  # Use frame timestamps
        "-q:v", "2",  # High quality output
        os.path.join(output_folder, "slide_%04d.png")
    ]
    
    try:
        # Run ffmpeg and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode != 0:
            print(f"FFmpeg completed with warnings:")
            # Only show relevant error lines
            for line in result.stderr.split('\n'):
                if "Error" in line or "failed" in line or "Invalid" in line:
                    print(f"  {line}")
        
        # Check if any frames were extracted
        frame_files = [f for f in os.listdir(output_folder) 
                      if f.startswith("slide_") and f.lower().endswith('.png')]
        
        if not frame_files:
            print("\nWarning: No frames extracted. Possible reasons:")
            print(f"  - Video may not contain clear slide transitions")
            print(f"  - Threshold ({threshold}) may be too high")
            print("Try rerunning with a lower threshold (e.g., --video-threshold 0.25)")
            return False
        
        print(f"\nSuccessfully extracted {len(frame_files)} frames to {output_folder}")
        return True
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return False

def analyze_embedding_distances(embeddings, sample_size=200):
    """Analyze embedding distances to understand similarity space"""
    n = len(embeddings)
    sample_size = min(sample_size, n)
    indices = np.random.choice(n, sample_size, replace=False)
    sample_embeddings = embeddings[indices]
    
    # Calculate pairwise distances
    distances = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            distances.append(np.linalg.norm(sample_embeddings[i] - sample_embeddings[j]))
    
    distances = np.array(distances)
    
    # K-distance graph for DBSCAN parameter estimation
    k = min(5, sample_size-1)
    k_distances = []
    for i in range(sample_size):
        dists = sorted([np.linalg.norm(sample_embeddings[i] - sample_embeddings[j]) 
                       for j in range(sample_size) if i != j])
        k_distances.append(dists[k-1])
    
    k_distances.sort()
    
    return {
        'min': np.min(distances),
        'max': np.max(distances),
        'mean': np.mean(distances),
        'median': np.median(distances),
        'std': np.std(distances),
        'percentile_75': np.percentile(distances, 75),
        'percentile_90': np.percentile(distances, 90),
        'k_distances': k_distances,
        'sample_size': sample_size
    }

def visualize_embeddings(embeddings, output_path="embedding_analysis.png"):
    """Create PCA visualization of embedding space"""
    try:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7)
        plt.title('Embedding Space (2D PCA Projection)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Warning: Could not create visualization: {str(e)}")
        return None

def generate_embeddings(image_folder, model_name="vit_large_patch16_224"):
    """Generate embeddings for PNG images"""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    image_paths = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder) 
        if f.lower().endswith('.png') and os.path.isfile(os.path.join(image_folder, f))
    ]
    
    if not image_paths:
        print(f"No PNG images found in {image_folder}")
        sys.exit(1)
    
    embeddings = []
    valid_paths = []
    
    print(f"Generating embeddings for {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                embedding = model(tensor)
                embeddings.append(embedding.squeeze().numpy())
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    return np.array(embeddings), valid_paths

def load_or_create_embeddings(image_folder, model_name="vit_large_patch16_224"):
    """Load cached embeddings or create new ones"""
    cache_dir = os.path.join(image_folder, ".slide_cluster_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    index_path = os.path.join(cache_dir, "embeddings.index")
    paths_path = os.path.join(cache_dir, "image_paths.json")
    
    # Load cached embeddings if valid
    if os.path.exists(index_path) and os.path.exists(paths_path):
        try:
            with open(paths_path, 'r') as f:
                saved_paths = json.load(f)
            
            valid_paths = [p for p in saved_paths if os.path.exists(p)]
            new_paths = [
                os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                if f.lower().endswith('.png') and os.path.join(image_folder, f) not in valid_paths
            ]
            
            if not new_paths:
                print("Using cached embeddings")
                index = faiss.read_index(index_path)
                embeddings = index.reconstruct_n(0, index.ntotal)
                return embeddings, valid_paths
        except Exception as e:
            print(f"Cache corrupted: {str(e)}")
    
    # Generate new embeddings
    embeddings, image_paths = generate_embeddings(image_folder, model_name)
    
    if len(embeddings) == 0:
        print("ERROR: No valid embeddings generated.")
        sys.exit(1)
    
    # Save to cache
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, index_path)
    with open(paths_path, 'w') as f:
        json.dump(image_paths, f)
    
    return embeddings, image_paths

def find_optimal_threshold(embeddings, min_cluster_size=2, max_cluster_size=20, max_threshold=None, step=0.05):
    """
    Find threshold where largest cluster size <= max_cluster_size
    
    CRITICAL CHANGE: The largest cluster size is defined as the maximum of:
    1. The largest DBSCAN cluster size
    2. The total number of noise points (treated as ONE cluster)
    
    This ensures we don't end up with a situation where we have many noise points
    that would form one enormous cluster if they were grouped together.
    
    CORRECTED LOGIC: We search through all thresholds to find the one that
    minimizes the largest cluster size, not just the first one that satisfies
    the constraint.
    """
    print("\n" + "="*50)
    print("EMBEDDING SPACE ANALYSIS")
    print("="*50)
    
    # Analyze embedding distances
    dist_stats = analyze_embedding_distances(embeddings)
    if dist_stats:
        print(f"\nEmbedding distance statistics (sample of {dist_stats['sample_size']} points):")
        print(f"  Min: {dist_stats['min']:.4f}, Max: {dist_stats['max']:.4f}")
        print(f"  Mean: {dist_stats['mean']:.4f}, Median: {dist_stats['median']:.4f}")
        print(f"  75th: {dist_stats['percentile_75']:.4f}, 90th: {dist_stats['percentile_90']:.4f}")
        
        # Set reasonable threshold range
        if max_threshold is None:
            max_threshold = min(1.5, dist_stats['percentile_90'] * 1.5)
    
    print(f"\nFinding threshold that minimizes largest cluster size...")
    print("Note: Largest cluster size = max(largest DBSCAN cluster, total noise points)")
    print("      (all noise points are treated as ONE cluster for this calculation)")
    
    # Start with low threshold and search through all possibilities
    threshold = 0.05
    best_threshold = threshold
    min_largest_cluster_size = float('inf')
    
    # We'll track results for reporting
    results = []
    
    # Start from low threshold and increase
    while threshold <= max_threshold:
        clustering = DBSCAN(eps=threshold, min_samples=min_cluster_size, metric='euclidean')
        labels = clustering.fit_predict(embeddings)
        
        # Count cluster sizes
        unique_labels = np.unique(labels)
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:  # Skip noise
                cluster_size = np.sum(labels == label)
                cluster_sizes.append(cluster_size)
        
        # Count noise points
        n_noise = np.sum(labels == -1)
        
        # CRITICAL CHANGE: The largest cluster size is the maximum of:
        # 1. The largest DBSCAN cluster
        # 2. The total number of noise points (treated as ONE cluster)
        largest_cluster_size = max(
            max(cluster_sizes) if cluster_sizes else 0,
            n_noise
        )
        
        # Store results
        results.append({
            'threshold': threshold,
            'largest_cluster': largest_cluster_size,
            'dbscan_clusters': len(cluster_sizes),
            'noise_points': n_noise,
            'total_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0) + n_noise
        })
        
        # Track the threshold with the smallest largest_cluster_size
        if largest_cluster_size < min_largest_cluster_size:
            min_largest_cluster_size = largest_cluster_size
            best_threshold = threshold
        
        # Increase threshold
        threshold += step
    
    # Print results
    print("\nThreshold search results:")
    print("{:<10} {:<20} {:<15} {:<15} {:<15}".format(
        "Threshold", "Largest Cluster Size", "DBSCAN Clusters", "Noise Points", "Total Clusters"))
    
    for res in results:
        print("{:<10.2f} {:<20} {:<15} {:<15} {:<15}".format(
            res['threshold'],
            res['largest_cluster'],
            res['dbscan_clusters'],
            res['noise_points'],
            res['total_clusters']
        ))
    
    # Check if we found a threshold that satisfies the constraint
    if min_largest_cluster_size <= max_cluster_size:
        print(f"\nSelected threshold: {best_threshold:.2f}")
        print(f"Largest cluster size at this threshold: {min_largest_cluster_size}")
    else:
        print(f"\nWarning: Could not find threshold where largest cluster size <= {max_cluster_size}")
        print(f"Using threshold {best_threshold:.2f} with smallest possible largest cluster size: {min_largest_cluster_size}")
    
    print(f"  (comprised of either the largest DBSCAN cluster or all noise points treated as one cluster)")
    
    return best_threshold

def cluster_images(embeddings, threshold=None, min_cluster_size=2, max_cluster_size=20):
    """Cluster images with noise points as individual clusters"""
    if threshold is None:
        threshold = find_optimal_threshold(embeddings, min_cluster_size, max_cluster_size)
    
    print(f"\n" + "="*50)
    print(f"CLUSTERING ANALYSIS (threshold={threshold:.2f})")
    print("="*50)
    
    clustering = DBSCAN(eps=threshold, min_samples=min_cluster_size, metric='euclidean')
    labels = clustering.fit_predict(embeddings)
    
    # Convert noise points (-1) to individual clusters
    unique_labels = np.unique(labels)
    n_dbscan_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    # Calculate largest cluster size (with noise points as one cluster)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
    largest_cluster_size = max(
        max(cluster_sizes) if cluster_sizes else 0,
        n_noise
    )
    
    # Reassign noise points to unique cluster IDs
    new_labels = labels.copy()
    if -1 in unique_labels:
        current_max_id = max([l for l in unique_labels if l != -1], default=-1)
        for i, is_noise in enumerate(labels == -1):
            if is_noise:
                current_max_id += 1
                new_labels[i] = current_max_id
    
    # Final cluster count
    final_clusters = len(np.unique(new_labels))
    
    # Check cluster sizes
    cluster_sizes = {}
    for label in np.unique(new_labels):
        cluster_sizes[label] = np.sum(new_labels == label)
    
    max_size = max(cluster_sizes.values()) if cluster_sizes else 0
    
    print(f"\nFinal clustering result:")
    print(f"  DBSCAN found {n_dbscan_clusters} clusters")
    print(f"  {n_noise} images not in clusters (each becomes separate cluster)")
    print(f"  Total clusters: {final_clusters}")
    print(f"  Largest cluster: {max_size} images")
    print(f"  Note: For threshold optimization, largest cluster size was calculated as {largest_cluster_size}")
    print(f"        (max of largest DBSCAN cluster and total noise points treated as one cluster)")
    
    return new_labels, threshold

def calculate_sharpness(image_path):
    """Calculate image sharpness using Laplacian variance"""
    img = cv2.imread(image_path)
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def stack_slide_images(image_paths, output_path):
    """Stack multiple slide images with alignment, with safeguards for large clusters"""
    if len(image_paths) < 2:
        shutil.copy2(image_paths[0], output_path)
        return True
    
    # For large clusters, skip alignment to prevent hanging
    if len(image_paths) > 15:
        print(f"  Skipping alignment for large cluster ({len(image_paths)} images), using simple stacking")
        
        # Read all valid images
        valid_images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                valid_images.append(img)
        
        if not valid_images:
            # Fallback to first image
            shutil.copy2(image_paths[0], output_path)
            return True
        
        # Stack using median without alignment
        stacked = np.median([img.astype(np.float32) for img in valid_images], axis=0).astype(np.uint8)
        cv2.imwrite(output_path, stacked)
        return True
    
    # Read and convert to grayscale for alignment
    images = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            valid_paths.append(path)
    
    if len(images) < 2:
        # Not enough valid images for stacking
        if valid_paths:
            shutil.copy2(valid_paths[0], output_path)
        return len(valid_paths) > 0
    
    # Align images
    reference = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    aligned_images = [images[0]]
    
    for i in range(1, len(images)):
        src = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        
        try:
            # Add timeout for ECC alignment (simple version without signals)
            start_time = time.time()
            
            _, warp_matrix = cv2.findTransformECC(
                src, reference, warp_matrix, cv2.MOTION_AFFINE, criteria
            )
            
            # Check if it took too long (should be fast for presentation slides)
            if time.time() - start_time > 5.0:
                print(f"  Warning: Image alignment took too long for image {i+1}, skipping alignment")
                aligned_images.append(images[i])
            else:
                aligned = cv2.warpAffine(
                    images[i], warp_matrix, (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                aligned_images.append(aligned)
        except Exception as e:
            print(f"  Warning: Could not align image {i+1}: {str(e)}")
            # Use the original image without alignment
            aligned_images.append(images[i])
    
    # Stack using median
    stacked = np.median([img.astype(np.float32) for img in aligned_images], axis=0).astype(np.uint8)
    cv2.imwrite(output_path, stacked)
    return True

def preprocess_for_ocr(image_path, output_path):
    """Preprocess image for optimal OCR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Error: Could not read image {image_path}")
            shutil.copy2(image_path, output_path)
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(thresh, -1, kernel)
        cv2.imwrite(output_path, sharpened)
        return True
    except Exception as e:
        print(f"Error in preprocessing {image_path}: {str(e)}")
        shutil.copy2(image_path, output_path)
        return False

def validate_and_split_clusters(image_paths, cluster_labels, max_cluster_size=20, sharpness_threshold=0.9):
    """
    Validate clusters and split if they contain different slides
    
    Checks if stacking images improves sharpness. If not, splits the cluster.
    """
    print("\n" + "="*50)
    print("VALIDATING AND SPLITTING CLUSTERS")
    print("="*50)
    
    # Group images by cluster
    clusters = {}
    for img_path, cluster_id in zip(image_paths, cluster_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(img_path)
    
    # Track new clusters
    new_clusters = {}
    cluster_id_counter = max(clusters.keys()) + 1
    
    # Validate each cluster
    for cluster_id, paths in clusters.items():
        if len(paths) <= 1:
            # Single-image clusters don't need validation
            new_clusters[cluster_id] = paths
            continue
        
        print(f"\nValidating cluster {cluster_id} ({len(paths)} images)...")
        
        # Calculate sharpness of individual images
        sharpness_scores = [(p, calculate_sharpness(p)) for p in paths]
        sharpness_values = [score for _, score in sharpness_scores]
        avg_sharpness = np.mean(sharpness_values)
        max_sharpness = max(sharpness_values)
        
        # Try stacking all images
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        stack_success = stack_slide_images(paths, temp_path)
        if stack_success:
            stacked_sharpness = calculate_sharpness(temp_path)
            os.unlink(temp_path)
            
            print(f"  Average individual sharpness: {avg_sharpness:.2f}")
            print(f"  Max individual sharpness: {max_sharpness:.2f}")
            print(f"  Stacked image sharpness: {stacked_sharpness:.2f}")
            
            # If stacking improves sharpness significantly, keep as one cluster
            if stacked_sharpness >= max_sharpness * sharpness_threshold:
                new_clusters[cluster_id] = paths
                print(f"  Cluster {cluster_id} validated (stacking improves sharpness)")
                continue
        
        # If we get here, the cluster needs splitting
        print(f"  Cluster {cluster_id} needs splitting (stacking doesn't improve sharpness)")
        
        # Sort images by sharpness (highest first)
        sharpness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Start with the sharpest image as the first sub-cluster
        current_cluster = [sharpness_scores[0][0]]
        sub_clusters = [current_cluster]
        
        # Try to add each subsequent image to an existing sub-cluster
        for img_path, sharpness in sharpness_scores[1:]:
            added = False
            
            # Try each existing sub-cluster
            for sub_cluster in sub_clusters:
                # Create a temporary stack with this image added
                test_cluster = sub_cluster + [img_path]
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    test_path = temp_file.name
                
                stack_slide_images(test_cluster, test_path)
                test_sharpness = calculate_sharpness(test_path)
                os.unlink(test_path)
                
                # If adding this image improves or maintains sharpness, add it to this sub-cluster
                if test_sharpness >= max([calculate_sharpness(p) for p in test_cluster]) * sharpness_threshold:
                    sub_cluster.append(img_path)
                    added = True
                    break
            
            # If not added to any existing sub-cluster, create a new one
            if not added:
                new_cluster = [img_path]
                sub_clusters.append(new_cluster)
        
        # Assign new cluster IDs
        for i, sub_cluster in enumerate(sub_clusters):
            if i == 0:
                # Keep the original cluster ID for the first sub-cluster
                new_clusters[cluster_id] = sub_cluster
            else:
                new_cluster_id = cluster_id_counter
                cluster_id_counter += 1
                new_clusters[new_cluster_id] = sub_cluster
                print(f"  Created new cluster {new_cluster_id} with {len(sub_cluster)} images")
    
    # Convert back to labels array
    new_labels = np.copy(cluster_labels)
    for new_id, paths in new_clusters.items():
        for path in paths:
            try:
                idx = image_paths.index(path)
                new_labels[idx] = new_id
            except ValueError:
                # Handle case where path might not be in image_paths (shouldn't happen)
                continue
    
    return new_labels

def organize_clusters(image_paths, cluster_labels, output_base="clusters"):
    """Organize images into clusters with lexically least filename as directory name"""
    shutil.rmtree(output_base, ignore_errors=True)
    os.makedirs(output_base, exist_ok=True)
    
    # Group by cluster
    clusters = {}
    for img_path, cluster_id in zip(image_paths, cluster_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(img_path)
    
    # Create directories
    cluster_dir_map = {}
    print(f"\n" + "="*50)
    print(f"ORGANIZING {len(image_paths)} IMAGES INTO {len(clusters)} CLUSTERS")
    print("="*50)
    
    # Show cluster size distribution
    sizes = [len(imgs) for imgs in clusters.values()]
    size_counts = {}
    for size in sizes:
        size_counts[size] = size_counts.get(size, 0) + 1
    
    print("\nCluster size distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"  {count} clusters with {size} image{'s' if size > 1 else ''}")
    
    for cluster_id, paths in clusters.items():
        # Use lexically least filename as directory name
        dir_name = min([os.path.splitext(os.path.basename(p))[0] for p in paths])
        counter = 1
        temp_name = dir_name
        while temp_name in cluster_dir_map.values():
            temp_name = f"{dir_name}_{counter}"
            counter += 1
        dir_name = temp_name
        
        cluster_dir = os.path.join(output_base, dir_name)
        cluster_dir_map[cluster_id] = dir_name
        os.makedirs(cluster_dir, exist_ok=True)
        
        for img_path in paths:
            try:
                shutil.move(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))
            except Exception as e:
                print(f"Error moving {img_path}: {str(e)}")
    
    return cluster_dir_map

def process_cluster_images(cluster_dir, output_path):
    """Process cluster images into single preprocessed image"""
    image_paths = [
        os.path.join(cluster_dir, f) for f in os.listdir(cluster_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_paths: 
        print(f"Warning: No images found in cluster {cluster_dir}")
        return False
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Stack images or use sharpest if single image
        if not stack_slide_images(image_paths, temp_path):
            print(f"  Warning: Stack failed for cluster {os.path.basename(cluster_dir)}")
            sharpness_scores = [(p, calculate_sharpness(p)) for p in image_paths]
            sharpest_path = max(sharpness_scores, key=lambda x: x[1])[0]
            shutil.copy2(sharpest_path, temp_path)
        
        # Preprocess for OCR
        if not preprocess_for_ocr(temp_path, output_path):
            print(f"  Warning: Preprocessing failed for cluster {os.path.basename(cluster_dir)}")
        
        os.unlink(temp_path)
        return True
    except Exception as e:
        print(f"Error processing cluster {cluster_dir}: {str(e)}")
        if image_paths:
            shutil.copy2(image_paths[0], output_path)
        return True if image_paths else False

def create_pdf_from_images(image_folder, output_pdf="presentation.pdf"):
    """Create PDF from processed images"""
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not image_files:
        print("No images found to create PDF")
        return None
    
    print(f"\n" + "="*50)
    print(f"CREATING PDF FROM {len(image_files)} IMAGES")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_files = []
        for img_path in image_files:
            pdf_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(img_path))[0] + ".pdf")
            try:
                subprocess.run(["convert", img_path, pdf_path], 
                              check=True, capture_output=True)
                pdf_files.append(pdf_path)
            except Exception as e:
                print(f"Error converting {img_path}: {str(e)}")
        
        if pdf_files:
            try:
                subprocess.run(["pdfunite"] + sorted(pdf_files) + [output_pdf], check=True)
                print(f"PDF created: {os.path.abspath(output_pdf)}")
                return output_pdf
            except Exception as e:
                print(f"Error creating final PDF: {str(e)}")
    return None

def add_ocr_to_pdf(input_pdf, output_pdf="ocr_presentation.pdf"):
    """Add OCR text layer to PDF using a more reliable image-based approach"""
    try:
        print(f"\n" + "="*50)
        print("ADDING OCR TEXT LAYER TO PDF")
        print("="*50)
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Converting PDF pages to high-quality images...")
            
            # Convert PDF to PNG images (300 DPI for good OCR quality)
            convert_cmd = [
                "pdftoppm", 
                "-png", 
                "-r", "300",
                input_pdf,
                os.path.join(temp_dir, "page")
            ]
            subprocess.run(convert_cmd, check=True)
            
            # Get list of generated images
            image_files = sorted(glob.glob(os.path.join(temp_dir, "page*.png")))
            print(f"Converted to {len(image_files)} images for OCR processing")
            
            # Process each image with Tesseract
            hocr_files = []
            for i, img_path in enumerate(tqdm(image_files, desc="Running OCR")):
                hocr_path = os.path.join(temp_dir, f"page_{i+1:04d}.hocr")
                hocr_files.append(hocr_path)
                
                # Run Tesseract in hOCR mode (HTML with OCR data)
                subprocess.run([
                    "tesseract", img_path, 
                    os.path.splitext(hocr_path)[0],
                    "hocr"
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Convert hOCR to searchable PDF
            print("Combining OCR results into searchable PDF...")
            subprocess.run([
                "hocr-pdf", 
                temp_dir,
                output_pdf
            ], check=True)
            
            print(f"OCR PDF created: {os.path.abspath(output_pdf)}")
            return output_pdf
            
    except Exception as e:
        print(f"Error adding OCR to PDF: {str(e)}")
        
        # Fallback approach if hocr-pdf isn't available
        try:
            print("Attempting fallback method with Tesseract's pdf renderer...")
            subprocess.run([
                "tesseract", input_pdf, 
                output_pdf.replace(".pdf", ""), 
                "pdf"
            ], check=True)
            return output_pdf
        except:
            print("Fallback method also failed. You may need to install additional tools:")
            print("  - Ubuntu/Debian: sudo apt-get install poppler-utils tesseract-ocr hocr-tools")
            print("  - macOS with Homebrew: brew install poppler tesseract hocr-tools")
            return None

def main(image_folder, video_path=None, video_threshold=0.35, threshold=None, max_cluster_size=20, model_name="vit_large_patch16_224"):
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Handle video extraction if specified
    if video_path:
        print(f"\nVideo processing requested: {video_path}")
        
        # Create a dedicated folder for video frames
        video_frames_folder = os.path.join(image_folder, "video_frames")
        
        # Extract frames from video
        if not extract_video_frames(video_path, video_frames_folder, threshold=video_threshold):
            print("Failed to extract frames from video. Exiting.")
            sys.exit(1)
        
        # Use the extracted frames as our image source
        image_folder = video_frames_folder
        print(f"Using extracted frames from {video_frames_folder} for processing")
    
    # Create output directories
    os.makedirs("processed_clusters", exist_ok=True)
    
    # Main processing pipeline
    print("\n" + "="*50)
    print("STEP 1: CLUSTERING IMAGES")
    print("="*50)
    
    embeddings, image_paths = load_or_create_embeddings(image_folder, model_name)
    vis_path = visualize_embeddings(embeddings)
    if vis_path: print(f"Embedding visualization: {vis_path}")
    
    # Initial clustering
    cluster_labels, final_threshold = cluster_images(embeddings, threshold, max_cluster_size=max_cluster_size)
    
    # Validate and possibly split clusters
    cluster_labels = validate_and_split_clusters(image_paths, cluster_labels, max_cluster_size)
    
    cluster_dir_map = organize_clusters(image_paths, cluster_labels)
    
    # Process clusters
    print("\n" + "="*50)
    print("STEP 2: PROCESSING CLUSTERS FOR BEST OCR")
    print("="*50)
    
    # Process clusters with progress bar
    processed_count = 0
    cluster_dirs = list(cluster_dir_map.values())
    
    # Add explicit check to ensure progress bar works
    print(f"Found {len(cluster_dirs)} clusters to process")
    if not cluster_dirs:
        print("ERROR: No clusters found to process. Check your input images and clustering parameters.")
        return None
    
    for dir_name in tqdm(cluster_dirs, desc="Processing clusters", total=len(cluster_dirs)):
        cluster_path = os.path.join("clusters", dir_name)
        output_path = os.path.join("processed_clusters", f"{dir_name}.png")
        if process_cluster_images(cluster_path, output_path):
            processed_count += 1
    
    print(f"\nSuccessfully processed {processed_count} out of {len(cluster_dirs)} clusters")
    
    # Create final PDF
    pdf_path = create_pdf_from_images("processed_clusters", "presentation.pdf")
    if pdf_path:
        ocr_pdf_path = add_ocr_to_pdf(pdf_path, "ocr_presentation.pdf")
        if ocr_pdf_path:
            print(f"\nPROCESS COMPLETED SUCCESSFULLY!")
            print(f"Final OCR-enhanced PDF: {os.path.abspath(ocr_pdf_path)}")
            return ocr_pdf_path
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process presentation slide images or extract from video')
    parser.add_argument('--folder', type=str, required=True, help='Folder for output and temporary files')
    parser.add_argument('--video', type=str, default=None, help='Video file to extract slides from')
    parser.add_argument('--video-threshold', type=float, default=0.35, 
                        help='Scene change threshold for video extraction (0-1, higher = less sensitive)')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='Clustering threshold (auto-optimized if not provided)')
    parser.add_argument('--max-cluster-size', type=int, default=20,
                        help='Maximum number of images allowed in a single cluster (default: 20)')
    parser.add_argument('--model', type=str, default='vit_large_patch16_224',
                        help='timm model for embeddings')
    
    args = parser.parse_args()
    
    # If video is specified, ensure ffmpeg is available
    if args.video and not check_ffmpeg():
        print("ERROR: ffmpeg is required for video processing but not found.")
        print("Please install ffmpeg before using the --video option.")
        sys.exit(1)
    
    main(args.folder, args.video, args.video_threshold, args.threshold, args.max_cluster_size, args.model)