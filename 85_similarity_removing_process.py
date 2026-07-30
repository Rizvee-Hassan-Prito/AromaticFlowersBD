
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_FOLDER = "./Aromatic Flowers"        # Base path containing all flower subfolders
TARGET_SUBFOLDERS = ["Single", "Bulk"]   # Subfolder names to search for
SIMILARITY_THRESHOLD = 0.85              # 0.95 = 95% similarity threshold
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
BATCH_SIZE = 32                          # Inference batch size

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. LOAD MOBILENETV3 MODEL
# ---------------------------------------------------------
print(f"Loading MobileNetV3 (ImageNet1K V2) on {device}...")

weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
model = models.mobilenet_v3_large(weights=weights)

# Remove the final classification layer to extract raw 1280-dim feature vectors
model.classifier[3] = nn.Identity()

model = model.to(device)
model.eval()

preprocess = weights.transforms()

# ---------------------------------------------------------
# 2. LOCATE TARGET SUBFOLDERS
# ---------------------------------------------------------
subfolder_paths = []
for root, dirs, _ in os.walk(ROOT_FOLDER):
    if os.path.basename(root) in TARGET_SUBFOLDERS:
        subfolder_paths.append(root)

print(f"Found {len(subfolder_paths)} target subfolders to process.")

# ---------------------------------------------------------
# 3. PROCESS AND AUTOMATICALLY DELETE DUPLICATES
# ---------------------------------------------------------
total_deleted = 0

for folder in subfolder_paths:
    print(f"\nScanning: {folder}")
    
    # Collect images
    image_paths = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    
    if len(image_paths) < 2:
        print(" -> Fewer than 2 images found. Skipping.")
        continue

    # Load and preprocess
    valid_paths = []
    tensor_list = []
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor_list.append(preprocess(img))
            valid_paths.append(path)
        except Exception as e:
            print(f" -> Skipping unreadable file {os.path.basename(path)}: {e}")

    if len(valid_paths) < 2:
        continue

    # Extract feature embeddings in batches
    all_embeddings = []
    with torch.no_grad():
        for b_idx in range(0, len(tensor_list), BATCH_SIZE):
            batch_tensors = torch.stack(tensor_list[b_idx:b_idx + BATCH_SIZE]).to(device)
            raw_embeddings = model(batch_tensors)
            all_embeddings.append(raw_embeddings)

    embeddings = torch.cat(all_embeddings, dim=0)
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)

    # Calculate Cosine Similarity Matrix
    similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.T).cpu()

    # Find and delete duplicate images
    deleted_paths = set()
    num_images = len(valid_paths)
    folder_deleted_count = 0

    for i in range(num_images):
        path_i = valid_paths[i]
        if path_i in deleted_paths:
            continue

        for j in range(i + 1, num_images):
            path_j = valid_paths[j]
            if path_j in deleted_paths:
                continue

            score = similarity_matrix[i][j].item()

            if score >= SIMILARITY_THRESHOLD:
                img1_name = os.path.basename(path_i)
                img2_name = os.path.basename(path_j)

                try:
                    os.remove(path_j)
                    deleted_paths.add(path_j)
                    folder_deleted_count += 1
                    total_deleted += 1
                    print(f"  [Deleted] {img2_name} ({score * 100:.1f}% match with {img1_name})")
                except Exception as e:
                    print(f"  [Error] Failed to delete {img2_name}: {e}")

    if folder_deleted_count == 0:
        print(" -> No duplicates found above threshold.")

print(f"\n==================================================")
print(f"Processing Complete! Total duplicate images deleted: {total_deleted}")
print(f"==================================================")




####################################
# For Visualization
####################################

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# from PIL import Image
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------
# FOLDER_PATH = "./Aromatic Flowers/Golap/Single"  # Replace with your folder path
# SIMILARITY_THRESHOLD = 0.85                  # 0.80 = 80% similarity
# VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# # Set device (GPU if available, otherwise CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------------------------------------------------------
# # 1. LOAD MOBILENETV3 MODEL
# # ---------------------------------------------------------
# print(f"Loading MobileNetV3 model on {device}...")

# # Load pre-trained MobileNetV3 Large weights
# weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
# model = models.mobilenet_v3_large(weights=weights)

# # Remove the final classification linear layer to get raw feature embeddings
# model.classifier[3] = nn.Identity()

# model = model.to(device)
# model.eval()

# # Get the recommended image transformation pipeline for this model
# preprocess = weights.transforms()

# # ---------------------------------------------------------
# # 2. COLLECT IMAGES
# # ---------------------------------------------------------
# image_paths = [
#     os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH)
#     if f.lower().endswith(VALID_EXTENSIONS)
# ]

# if not image_paths:
#     print("No valid images found in the specified folder.")
#     exit()

# print(f"Found {len(image_paths)} images. Loading and processing...")

# # ---------------------------------------------------------
# # 3. GENERATE EMBEDDINGS WITH MOBILENETV3
# # ---------------------------------------------------------
# loaded_images = []
# valid_paths = []
# tensor_batch = []

# for path in image_paths:
#     try:
#         img = Image.open(path).convert("RGB")
#         loaded_images.append(img)
#         valid_paths.append(path)
        
#         # Preprocess image into a PyTorch tensor
#         tensor_img = preprocess(img)
#         tensor_batch.append(tensor_img)
#     except Exception as e:
#         print(f"Skipping unreadable file {path}: {e}")

# # Stack tensors into a single batch and pass through the model
# input_tensors = torch.stack(tensor_batch).to(device)

# with torch.no_grad():
#     # Extract feature embeddings
#     raw_embeddings = model(input_tensors)
#     # L2 normalize embeddings for cosine similarity computation
#     norm_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

# # ---------------------------------------------------------
# # 4. COMPUTE COSINE SIMILARITY MATRIX
# # ---------------------------------------------------------
# # Matrix multiplication on normalized vectors yields cosine similarity
# similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.T).cpu()

# # ---------------------------------------------------------
# # 5. DISPLAY MATCHES IN NOTEBOOK
# # ---------------------------------------------------------
# print(f"\n=== Image Pairs with >= {int(SIMILARITY_THRESHOLD * 100)}% Similarity ===\n")
# match_count = 0

# num_images = len(valid_paths)
# for i in range(num_images):
#     for j in range(i + 1, num_images):
#         score = similarity_matrix[i][j].item()
        
#         if score >= SIMILARITY_THRESHOLD:
#             match_count += 1
#             img1_name = os.path.basename(valid_paths[i])
#             img2_name = os.path.basename(valid_paths[j])
            
#             # Create side-by-side plot for the pair
#             fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            
#             axes[0].imshow(loaded_images[i])
#             axes[0].set_title(f"Image A:\n{img1_name}", fontsize=9)
#             axes[0].axis('off')
            
#             axes[1].imshow(loaded_images[j])
#             axes[1].set_title(f"Image B:\n{img2_name}", fontsize=9)
#             axes[1].axis('off')
            
#             plt.suptitle(f"Match #{match_count} — Similarity: {score * 100:.1f}%", fontsize=12, fontweight='bold')
#             plt.tight_layout()
#             plt.show()

# if match_count == 0:
#     print(f"No image pairs found above the {int(SIMILARITY_THRESHOLD * 100)}% threshold.")
# else:
#     print(f"\nTotal similar pairs displayed: {match_count}")
