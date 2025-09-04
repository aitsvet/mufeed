import os
import numpy as np
import sys
import timm
from PIL import Image
import faiss
import json
from pathlib import Path

image_folder = sys.argv[1]
image_paths = sorted([
    str(p) for p in Path(image_folder).glob("*.png")
    if p.is_file()
])
if not image_paths:
    print(f"Error: No PNG images found in {image_folder}", file=sys.stderr)
    sys.exit(1)

output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

model_name = sys.argv[3] if len(sys.argv) > 3 else "vit_large_patch16_224"
model = timm.create_model(model_name, pretrained=True, num_classes=0)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

embeddings = []
for img_path in image_paths:
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        embedding = model(tensor).squeeze().detach().numpy()
        embeddings.append(embedding)
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
embeddings = np.array(embeddings)

index_path = os.path.join(output_dir, "embeddings.index")
metadata_path = os.path.join(output_dir, "metadata.json")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
faiss.write_index(index, index_path)

metadata = {
    "image_paths": image_paths,
    "embedding_shape": embeddings.shape
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f)

print(f"Saved {len(image_paths)} embeddings to {output_dir}")