
# Device settings
cuda_devices="0"  

# Paths
# NOTE: Paths are relative to the 'scripts' directory where the test script is run
lang_encoder_path="../checkpoints/Llama-2-7b-chat-hf"
tokenizer_path="../checkpoints/Llama-2-7b-chat-hf"
pretrained_visual_encoder="../checkpoints/RadFM_vit3d.pth"
pretrained_adapter="../checkpoints/RadFM_perceiver_fc.pth"

# Trained model checkpoint (epoch 8)
ckpt_path="../outputs/Reg2RG_radgenome/checkpoint-8/pytorch_model.bin"

# Test Dataset Paths
data_folder='../src/Dataset/RadGenome-ChestCT/dataset/valid_preprocessed'
mask_folder='../src/Dataset/RadGenome-ChestCT/dataset/valid_region_mask'
report_file='../src/Dataset/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv'

# Cache dir
monai_cache_dir='../cache'

# Output path for generated reports
result_path='../results/radgenome_test_results.csv'
