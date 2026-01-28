import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="../checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(
        default="../checkpoints/Llama-2-7b-chat-hf",
        metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
        default="../checkpoints/RadFM_vit3d.pth")
    pretrained_adapter: Optional[str] = field(
        default="../checkpoints/RadFM_perceiver_fc.pth")

@dataclass
class DataArguments:
    # Training dataset paths
    data_folder: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/train_preprocessed')
    mask_folder: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/train_region_mask')
    report_file: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv')
    
    # Validation dataset paths (for evaluation during training)
    val_data_folder: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/valid_preprocessed')
    val_mask_folder: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/valid_region_mask')
    val_report_file: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    
    monai_cache_dir: Optional[str] = field(default='../cache')
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(
        default="../outputs/Reg2RG_radgenome")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pin_memory: bool = field(default=True)
