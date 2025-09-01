import os
import numpy as np
import torch
import timm
from PIL import Image
import faiss
from tqdm import tqdm
import json
from pathlib import Path
import argparse

def generate_embeddings(image_folder, model_name="vit_large_patch16_224"):
    """Generate embeddings for images in folder"""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    image_paths = sorted([
        str(p) for p in Path(image_folder).glob("*.png")
        if p.is_file()
    ])

    if not image_paths:
        raise ValueError(f"No PNG images found in {image_folder}")

    embeddings = []
    print(f"Generating embeddings for {len(image_paths)} images...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                embedding = model(tensor).squeeze().numpy()
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    return np.array(embeddings), image_paths

def save_embeddings(embeddings, image_paths, output_dir):
    """Save embeddings and metadata to files"""
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "embeddings.index")
    metadata_path = os.path.join(output_dir, "metadata.json")

    # Save Faiss index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, index_path)

    # Save metadata
    metadata = {
        "image_paths": image_paths,
        "embedding_shape": embeddings.shape
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved {len(image_paths)} embeddings to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image embeddings')
    parser.add_argument('--input', type=str, required=True, help='Folder with input images')
    parser.add_argument('--output', type=str, required=True, help='Folder to save embeddings')
    parser.add_argument('--model', type=str, default='vit_large_patch16_224', help='timm model name')

    args = parser.parse_args()

    try:
        embeddings, image_paths = generate_embeddings(args.input, args.model)
        save_embeddings(embeddings, image_paths, args.output)
    except Exception as e:
        print(f"Error in embedding generation: {str(e)}")
        sys.exit(1)