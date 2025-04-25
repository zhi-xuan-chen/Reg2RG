import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
import time
import argparse as ap

from model_trainer import ModelTrainer
from classifier import RadBertClassifier
from dataset import CTDataset
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

# 定义区域列表
regions = [
    'abdomen',
    'bone',
    'breast',
    'heart',
    'esophagus',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie'
]

# 基础路径和配置
base_csv_path = "/home/chenzhixuan/Workspace/FineGrainedCTRG/results/regionRG-radfm_radgenome_mask-crop-regions-image_remain_wrong"
gt_report_column = 'GT_combined_report'
pred_report_column = 'Pred_combined_report'
gpu_id = 0
model_path = "/jhcnas5/chenzhixuan/checkpoints/CT-CLIP/RadBertClassifier.pth"

label_cols = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
              'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
              'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
              'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
              'Interlobular septal thickening']
num_labels = len(label_cols)
print('Label columns: ', label_cols)

device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
print("Using ", device)

model = RadBertClassifier(n_classes=num_labels)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
print(model.eval())

# 设置优化器
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
scheduler = None

# 数据加载器配置
max_length = 512
num_workers = 8
batch_size = 2

all_results = []

for region in regions:
    csv_path = os.path.join(base_csv_path, f"{region}_reports.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist. Skipping...")
        continue

    save_path = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)
    print(f'\nNumber of test data in {region}: ', len(df))

    # 定义数据加载器
    dataloaders = {}

    ############### GT Report Inference ################
    test_data = CTDataset(df, num_labels, label_cols, gt_report_column, max_length, infer=True)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=True)
    dataloaders['test'] = test_dataloader

    epochs = 0
    trainer = ModelTrainer(model,
                           dataloaders,
                           num_labels,
                           epochs,
                           optimizer,
                           scheduler,
                           device,
                           save_path,
                           label_cols)
    start = time.time()
    print(f'----------------------Starting {region} GT Report Inferring----------------------')
    GT_predicted_labels = trainer.infer()

    finish = time.time()
    print('---------------------------------------------------------------')
    print(f'{region} GT Report Inferring Complete')
    print('Infer time: ', finish - start)

    ############### Pred Report Inference ################
    test_data = CTDataset(df, num_labels, label_cols, pred_report_column, max_length, infer=True)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=True)
    dataloaders['test'] = test_dataloader

    epochs = 0
    trainer = ModelTrainer(model,
                           dataloaders,
                           num_labels,
                           epochs,
                           optimizer,
                           scheduler,
                           device,
                           save_path,
                           label_cols)
    start = time.time()
    print(f'----------------------Starting {region} Pred Report Inferring----------------------')
    Pred_predicted_labels = trainer.infer()

    finish = time.time()
    print('---------------------------------------------------------------')
    print(f'{region} Pred Report Inferring Complete')
    print('Infer time: ', finish - start)

    ############################# evaluation #############################
    y_true = GT_predicted_labels
    y_pred = Pred_predicted_labels

    clf_report = classification_report(y_true, y_pred, target_names=label_cols, output_dict=True)
    macro_avg = clf_report['macro avg']

    result = {
        'region': region,
        'precision': macro_avg['precision'],
        'recall': macro_avg['recall'],
        'f1-score': macro_avg['f1-score'],
        'support': macro_avg['support']
    }
    all_results.append(result)

# 将所有区域的结果保存到一个 CSV 文件中
results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(base_csv_path, 'all_regions_macro_avg_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"All regions' macro avg results saved to {results_csv_path}")
    
#     micro_avg = clf_report['micro avg']

#     result = {
#         'region': region,
#         'precision': micro_avg['precision'],
#         'recall': micro_avg['recall'],
#         'f1-score': micro_avg['f1-score'],
#         'support': micro_avg['support']
#     }
#     all_results.append(result)

# # 将所有区域的结果保存到一个 CSV 文件中
# results_df = pd.DataFrame(all_results)
# results_csv_path = os.path.join(base_csv_path, 'all_regions_micro_avg_results.csv')
# results_df.to_csv(results_csv_path, index=False)
# print(f"All regions' micro avg results saved to {results_csv_path}")