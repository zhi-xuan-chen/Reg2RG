from huggingface_hub import snapshot_download

repo_id = "RadGenome/RadGenome-ChestCT"
local_dir = "./src/Dataset/RadGenome-ChestCT"

allow_patterns = []

# ---------------- FULL DOWNLOAD ----------------
allow_patterns.extend([
    "dataset/qa_templates/**",
    "dataset/radgenome_files/**",
    "dataset/report_hierarchy_tree.zip",
    "dataset/train_anatomy_mask_*",
    "dataset/train_region_mask.tar.gz",
    "dataset/valid_anatomy_mask.tar.gz",
    "dataset/valid_region_mask.tar.gz",
])

# ---------------- PARTIAL DOWNLOAD ----------------
# train_preprocessed: train_1 -> train_50
for i in range(1, 51):
    allow_patterns.append(
        f"dataset/train_preprocessed/train_{i}/**"
    )

# valid_preprocessed: valid_1 -> valid_10
for i in range(1, 11):
    allow_patterns.append(
        f"dataset/valid_preprocessed/valid_{i}/**"
    )

# ---------------- DOWNLOAD ----------------
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
)

print("✅ Download completed successfully!")
