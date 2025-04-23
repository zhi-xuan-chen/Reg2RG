import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer
from dataclasses import dataclass, field
from Model.Reg2RG import Reg2RG
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
from args.train_radgenome.jhcpu7 import ModelArguments, DataArguments, TrainingArguments
import numpy as np
import torch              
import random

REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        lang_xs, vision_xs, mask_xs, region2areas, attention_masks, labels = tuple([instance[key] for instance in instances] for key in ('lang_x', 'vision_x', 'mask_x', 'region2area', 'attention_mask', 'label'))
        
        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim = 0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim = 0)
        
        vision_temp = {area: [] for area in REGIONS}
        mask_temp = {area: [] for area in REGIONS}
        # get the shape of the vision tensor
        vision_shape = next(iter(vision_xs[0].values())).shape
        mask_shape = next(iter(mask_xs[0].values())).shape

        useless_regions = []
        
        for area in REGIONS:
            flag = False
            for i in range(len(vision_xs)):
                if area in vision_xs[i]:
                    vision_temp[area].append(vision_xs[i][area])
                    mask_temp[area].append(mask_xs[i][area])
                    flag = True
                else:
                    vision_temp[area].append(torch.zeros(vision_shape))
                    mask_temp[area].append(torch.zeros(mask_shape))
            if not flag:
                useless_regions.append(area)

        images = torch.cat([vision['image'].unsqueeze(0) for vision in vision_xs], dim = 0)
        
        # drop the useless regions from vision_temp
        for area in useless_regions:
            vision_temp.pop(area)
            mask_temp.pop(area)
        useful_regions = list(vision_temp.keys())
    
        vision_xs = {area: torch.cat([_.unsqueeze(0) for _ in vision_temp[area]], dim = 0) for area in useful_regions}
        # add image 
        vision_xs['image'] = images

        mask_xs = {area: torch.cat([_.unsqueeze(0) for _ in mask_temp[area]], dim = 0) for area in useful_regions}
        
        # print(vision_xs.shape,vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            mask_x=mask_xs,
            region2area = region2areas,
            attention_mask=attention_masks,
            labels = labels,
        )
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
def main():
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    Train_dataset = RadGenomeDataset_Train(
        text_tokenizer=model_args.tokenizer_path,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        cache_dir=data_args.monai_cache_dir,
    )

    # loader = torch.utils.data.DataLoader(
    #     Train_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=DataCollator())
    # for batch in tqdm.tqdm(loader):
    #     pass

    print("Setup Model")
    model = Reg2RG(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
    )

    # loader = torch.utils.data.DataLoader(
    #     Train_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=DataCollator())
    # for batch in tqdm.tqdm(loader):
    #     model(**batch)
    #     pass
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      args = training_args,
                      data_collator=DataCollator(),
                      )

    trainer.train()
    trainer.save_state()
      
if __name__ == "__main__":
    main()