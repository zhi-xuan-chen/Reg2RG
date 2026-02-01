import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="../checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="../checkpoints/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
        default="../checkpoints/RadFM_vit3d.pth")
    pretrained_adapter: Optional[str] = field(
        default="../checkpoints/RadFM_perceiver_fc.pth")
    # Use pretrained checkpoint from HuggingFace as fallback
    # TODO: Replace with your trained checkpoint once it's saved properly
    ckpt_path: Optional[str] = field(   
        default="../checkpoints/Reg2RG/pytorch_model.bin")

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/valid_preprocessed')
    mask_folder: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/valid_region_mask')
    report_file: Optional[str] = field(default='../src/Dataset/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    monai_cache_dir: Optional[str] = field(default='../cache')
    result_path: Optional[str] = field(default='../results/Reg2RG_radgenome/radgenome_test_predictions.csv')
