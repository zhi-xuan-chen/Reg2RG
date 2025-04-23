import tqdm.auto as tqdm
import os
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RegionRG_MaskEncoder_best.multimodality_model import RegionLLM
from Dataset.radgenome_dataset_test import RadGenomeDataset_Combined_Region_Image_Mask_Test
# from args.test_combined_region_radgenome.superpod import ModelArguments, DataArguments
from args.test_combined_region_radgenome.jhcpu1 import ModelArguments, DataArguments
import torch
from torch.utils.data import DataLoader
from safetensors import safe_open
import random
import numpy as np
import pandas as pd

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
        acc_nums, lang_xs, vision_xs, mask_xs, region2areas, questions, gt_combined_reports = tuple([instance[key] for instance in instances] for key in ('acc_num', 'lang_x','vision_x', 'mask_x', 'region2area', 'question', 'gt_combined_report'))
        
        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim = 0)
        
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
            acc_num=acc_nums,
            lang_x=lang_xs,
            vision_x=vision_xs,
            mask_x = mask_xs,
            region2area = region2areas,
            question = questions,
            gt_combined_report = gt_combined_reports,
        )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
# 预处理数据

def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    (model_args, data_args) = parser.parse_args_into_dataclasses()
    print(model_args.ckpt_path)
    
    # 判断结果保存路径是否存在，不存在则创建
    if os.path.exists(data_args.result_path):
        df = pd.read_csv(data_args.result_path)
        inferenced_id = df["AccNum"].tolist()
    else:
        # 创建一个空的DataFrame
        df = pd.DataFrame(columns=["AccNum", "Question", "GT_whole_report", "Pred_whole_report"])
        df.to_csv(data_args.result_path, index=False)
        inferenced_id = []

    print("Setup Data")
    Test_dataset = RadGenomeDataset_Combined_Region_Image_Mask_Test(
        text_tokenizer=model_args.tokenizer_path,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        wrong_path=data_args.wrong_path,
        cache_dir=data_args.monai_cache_dir,
        inferenced_id = inferenced_id
    )

    Test_dataloader = DataLoader(
        Test_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=DataCollator(),
        drop_last=False,
    )

    print("Setup Model")

    model = RegionLLM(
        text_tokenizer_path=model_args.tokenizer_path,
        lang_model_path=model_args.lang_encoder_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
    )

    # for batch in tqdm.tqdm(Test_dataloader):
    #     # 将张量保存为 NIfTI 文件
    #     model.generate(batch["lang_x"], batch["vision_x"], batch["region2area"])
    #     pass

    ckpt = torch.load(model_args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    print("load ckpt")
    model = model.cuda()
    print("model to cuda")

    model.eval()

    for sample in tqdm.tqdm(Test_dataloader):
        acc_num = sample["acc_num"][0]
        question = sample["question"][0]
        lang_x = sample["lang_x"].cuda()
        attention_mask = torch.ones_like(lang_x).cuda()
        # 循环遍历每个区域，移动到GPU上
        vision_x = {area: _.cuda() for area, _ in sample["vision_x"].items()}
        mask_x = {area: _.cuda() for area, _ in sample["mask_x"].items()}
        region2area = sample["region2area"]
        gt_combined_report = sample["gt_combined_report"][0]
        
        pred_combined_reports = model.generate(lang_x, attention_mask, vision_x, mask_x, region2area)
        pred_combined_report = pred_combined_reports[0]

        print('AccNum: ', acc_num)
        print('GT_report: ', gt_combined_report)
        print('Pred_report: ', pred_combined_report)
        
        # 将数据添加到DataFrame中并保存到CSV文件
        new_data = pd.DataFrame([[acc_num, question, gt_combined_report, pred_combined_report]], 
                                columns=["AccNum", "Question", "GT_whole_report", "Pred_whole_report"])
        new_data.to_csv(data_args.result_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    main()
