
# Experiment settings
experiment_name="Reg2RG_radgenome"
bf16=True

# Device settings
cuda_devices="1,5"  

# Torchrun settings
master_port=25368

# Paths
lang_encoder_path="/jhcnas5/chenzhixuan/checkpoints/Llama-2-7b-chat-hf"
tokenizer_path="/jhcnas5/chenzhixuan/checkpoints/Llama-2-7b-chat-hf"
pretrained_visual_encoder="/jhcnas5/chenzhixuan/checkpoints/LLM4CTRG/RadFM_vit3d.pth"
pretrained_adapter="/jhcnas5/chenzhixuan/checkpoints/LLM4CTRG/RadFM_perceiver_fc.pth"
data_folder='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/train_preprocessed'
mask_folder='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/train_region_mask'
report_file='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv'
wrong_path='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/processed_code/wrong_files/train_wrong_cases.json'
monai_cache_dir='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache' # useless
output_dir="/jhcnas5/chenzhixuan/checkpoints/FineGrainedCTRG/outputs/$experiment_name"
deepspeed_config="/home/chenzhixuan/Workspace/FineGrainedCTRG/ds_configs/stage2.json"

# Training settings
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=8
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=3
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
logging_steps=1
