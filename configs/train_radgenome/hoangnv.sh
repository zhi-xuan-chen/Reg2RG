
# Experiment settings
experiment_name="Reg2RG_radgenome"
bf16=False
fp16=True

# Device settings
cuda_devices="0"  

# Torchrun settings
master_port=29600

# Paths
# NOTE: Paths are relative to the 'scripts' directory where the training launching script is run
# Assuming code structure:
# Reg2RG/
#   scripts/ (run from here)
#   checkpoints/
#   src/
#   outputs/
#   cache/

lang_encoder_path="../checkpoints/Llama-2-7b-chat-hf"
tokenizer_path="../checkpoints/Llama-2-7b-chat-hf"

pretrained_visual_encoder="../checkpoints/RadFM_vit3d.pth"
pretrained_adapter="../checkpoints/RadFM_perceiver_fc.pth"

# Training Dataset Paths
data_folder='../src/Dataset/RadGenome-ChestCT/dataset/train_preprocessed'
mask_folder='../src/Dataset/RadGenome-ChestCT/dataset/train_region_mask'
report_file='../src/Dataset/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv'

# Validation Dataset Paths (for evaluation during training)
val_data_folder='../src/Dataset/RadGenome-ChestCT/dataset/valid_preprocessed'
val_mask_folder='../src/Dataset/RadGenome-ChestCT/dataset/valid_region_mask'
val_report_file='../src/Dataset/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv'

monai_cache_dir='../cache'
output_dir="../outputs/$experiment_name"
deepspeed_config="../ds_configs/stage2.json"

# Training settings
learning_rate=5e-5
per_device_train_batch_size=1
per_device_eval_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=8

# Evaluation settings - Run evaluation every epoch
evaluation_strategy="epoch"
eval_steps=1  # Not used when evaluation_strategy="epoch", but kept for reference

# Save settings - Save checkpoints and keep best based on validation loss
save_strategy="epoch"
save_steps=1  # Not used when save_strategy="epoch"
save_total_limit=3  # Keep last 3 checkpoints to save disk space
load_best_model_at_end=True  # Load best checkpoint at the end of training
metric_for_best_model="loss"  # Use validation loss as metric
greater_is_better=False  # Lower loss is better

# Optimizer settings
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="constant_with_warmup"

# Data loading
dataloader_num_workers=4

# Logging
logging_steps=1
logging_strategy="steps"
