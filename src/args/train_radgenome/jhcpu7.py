import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/checkpoints/LLM4CTRG/RadFM_vit3d.pth")
    pretrained_adapter: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/checkpoints/LLM4CTRG/RadFM_perceiver_fc.pth")

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_preprocessed')
    mask_folder: Optional[str] = field(default='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_region_mask')
    report_file: Optional[str] = field(default='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    wrong_path: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/processed_code/wrong_files/valid_wrong_cases.json')
    monai_cache_dir: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache')
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/checkpoints/FineGrainedCTRG/outputs")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pin_memory: bool = field(default=True)