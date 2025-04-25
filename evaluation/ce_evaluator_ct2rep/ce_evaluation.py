import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
import time
import argparse as ap

from model_trainer import ModelTrainer
from classifier import RadBertClassifier
from dataset import CTDataset
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

############################# label inference #############################
# path
csv_path = "/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports_rm_region_text.csv"
gt_report_column = 'GT_combined_report'
pred_report_column = 'Pred_combined_report'
gpu_id = 0
# extract the dir path from the csv_path
save_path = os.path.dirname(csv_path)
# model_path = "/jhcnas5/chenzhixuan/checkpoints/CT-CLIP/RadBertClassifier.pth"
model_path = "/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadBertClassifier.pth"

# load the csv file need to be evaluated
df = pd.read_csv(csv_path) 

label_cols = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
num_labels = len(label_cols)
print('Label columns: ', label_cols)
print('\nNumber of test data: ',len(df))

device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
print("Using ",device)

model = RadBertClassifier(n_classes=num_labels)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
print(model.eval())

#No need to set up for inference
# setting custom optimization parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
scheduler = None
# optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization

# Create dataloader
dataloaders = {}

max_length = 512
num_workers = 8
batch_size = 2

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
print('----------------------Starting GT Report Inferring----------------------')
GT_predicted_labels = trainer.infer()

finish = time.time()
print('---------------------------------------------------------------')
print('GT Report Inferring Complete')
print('Infer time: ',finish-start)

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
print('----------------------Starting Pred Report Inferring----------------------')
Pred_predicted_labels = trainer.infer()

finish = time.time()
print('---------------------------------------------------------------')
print('Pred Report Inferring Complete')
print('Infer time: ',finish-start)

label_columns = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
# add GT prefix
GT_label_columns = ['GT_'+col for col in label_columns]
# add Pred prefix
Pred_label_columns = ['Pred_'+col for col in label_columns]

inferred_data = pd.DataFrame()
inferred_data['AccNum'] = df['AccNum']
inferred_data['Question'] = df[gt_report_column]
inferred_data[gt_report_column] = df[gt_report_column]
inferred_data[pred_report_column] = df[pred_report_column]

for col,i in zip(GT_label_columns,range(num_labels)):
    inferred_data[col] = GT_predicted_labels[:,i]
    
for col,i in zip(Pred_label_columns,range(num_labels)):
    inferred_data[col] = Pred_predicted_labels[:,i]

save_df = os.path.join(save_path,'label_inferred.csv')
inferred_data.to_csv(save_df,index=False)
print('Inferred data saved to: ',save_df)

############################# evaluation #############################

y_true = GT_predicted_labels
y_pred = Pred_predicted_labels

cm = multilabel_confusion_matrix(y_true, y_pred)
clf = classification_report(y_true, y_pred, target_names=label_columns)

with open(os.path.join(save_path, 'test_classification_report.txt'), 'w') as file:
  file.write(clf)
  
# manual calculation:

# Initialize lists to store metrics
precision_list = []
recall_list = []
f1_list = []
support_list = []

# Calculate metrics for each label
for matrix in cm:
    TN, FP, FN, TP = matrix.ravel()
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    support = TP + FN

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    support_list.append(support)

# Calculate weighted averages
total_support = np.sum(support_list)
weighted_precision = np.sum([precision * support for precision, support in zip(precision_list, support_list)]) / total_support
weighted_recall = np.sum([recall * support for recall, support in zip(recall_list, support_list)]) / total_support
weighted_f1 = np.sum([f1 * support for f1, support in zip(f1_list, support_list)]) / total_support

# Create a DataFrame to save metrics
metrics_df = pd.DataFrame({
    'Label': label_columns,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1 Score': f1_list,
    'Support': support_list
})

# Add weighted averages as the last row
metrics_df.loc['Weighted Average'] = ['Weighted Average', weighted_precision, weighted_recall, weighted_f1, total_support]

# Save to CSV
metrics_df.to_csv(os.path.join(save_path, 'metrics_manual.csv'), index=False)



