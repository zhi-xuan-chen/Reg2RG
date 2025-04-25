
# Device settings
cuda_devices="6"  

# Paths
lang_encoder_path="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf"
tokenizer_path="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf"
pretrained_visual_encoder="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_vit3d.pth"
pretrained_adapter="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_perceiver_fc.pth"
ckpt_path="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/pytorch_model.bin"
data_folder='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_preprocessed'
mask_folder='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_region_mask'
report_file='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv'
result_path='/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports.csv'