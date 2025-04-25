import re
import pandas as pd
import numpy as np
import os

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
    'trachea',
]

def extract_region_area_dict(text):
    # 使用正则表达式查找所有区域及其名称
    pattern = r"The region (\d+) is (\w+)"
    matches = re.findall(pattern, text)

    # 创建字典来存储区域和名称
    region_dict = {f"region {num}": name for num, name in matches}

    return region_dict  

results_path = "/home/zchenhi/Workspace/FineGrainedCTRG/results/regionRG-radfm_ctrg_mask-crop-regions-image.csv"
df = pd.read_csv(results_path)

gt_combined_reports = df["GT_whole_report"].tolist()
pred_combined_reports = df["Pred_whole_report"].tolist()
accnums = df["AccNum"].tolist()

# 提取区域及其描述
gt_region_descriptions = [extract_region_area_dict(report) for report in gt_combined_reports]
pred_region_descriptions = [extract_region_area_dict(report) for report in pred_combined_reports]

####################################### 计算每个 sample 的准确率 #############################################
accuracies = {}
for accnum, gt_regions, pred_regions in zip(accnums, gt_region_descriptions, pred_region_descriptions):
    correct_predictions = 0
    total_predictions = 0
    for region, area in gt_regions.items():
        total_predictions += 1
        if pred_regions.get(region) == area:
            correct_predictions += 1
    acc = correct_predictions / total_predictions
    accuracies[accnum] = acc

# 计算所有 sample 的平均准确率和标准差
mean_acc = np.mean(list(accuracies.values()))
std_acc = np.std(list(accuracies.values()))

print("\n")
print("Accuracy for each sample:")
print(f"Mean accuracy: {mean_acc}")
print(f"Standard deviation: {std_acc}")
print("\n")

####################################### 计算每个区域所有 sample 的准确率 #############################################
# 计算 precision 和 recall
region_TP = {region: 0 for region in REGIONS}
region_FP = {region: 0 for region in REGIONS}
region_FN = {region: 0 for region in REGIONS}

# 初始化每个区域的字典用来存储FP和FN的accnum
region_FP_accnums = {region: [] for region in REGIONS}
region_FN_accnums = {region: [] for region in REGIONS}
for accnum, gt_regions, pred_regions in zip(accnums, gt_region_descriptions, pred_region_descriptions):
    for region, area in gt_regions.items():
        if pred_regions.get(region) == area:
            region_TP[area] += 1
        elif pred_regions.get(region) is None or pred_regions.get(region) not in REGIONS:
            region_FN[area] += 1
            region_FN_accnums[area].append(accnum)
        else:
            region_FP[pred_regions.get(region)] += 1
            region_FN[area] += 1
            region_FP_accnums[pred_regions.get(region)].append(accnum)
            region_FN_accnums[area].append(accnum)

region_precision = {region: region_TP[region] / (region_TP[region] + region_FP[region] + 1e-5) for region in REGIONS}
region_recall = {region: region_TP[region] / (region_TP[region] + region_FN[region] + 1e-5) for region in REGIONS}
# 计算 F1 score
region_f1 = {region: 2 * region_precision[region] * region_recall[region] / (region_precision[region] + region_recall[region] + 1e-5) for region in REGIONS}

# 输出每个区域的 precision, recall 和 F1 score
print("Precision, recall and F1 score for each region:")
for region in REGIONS:
    print(f"{region}:")
    print(f"Precision: {region_precision[region]}")
    print(f"Recall: {region_recall[region]}")
    print(f"F1 score: {region_f1[region]}")
    print("\n")
    
# 计算除了 lung 和 pleura 之外所有区域的平均 precision, recall 和 F1 score
regions_to_exclude = ['lung', 'pleura']
filtered_regions = [region for region in REGIONS if region not in regions_to_exclude]

filtered_precision = [region_precision[region] for region in filtered_regions]
filtered_recall = [region_recall[region] for region in filtered_regions]
filtered_f1 = [region_f1[region] for region in filtered_regions]

mean_region_precision = np.mean(filtered_precision)
std_region_precision = np.std(filtered_precision)
mean_region_recall = np.mean(filtered_recall)
std_region_recall = np.std(filtered_recall)
mean_region_f1 = np.mean(filtered_f1)
std_region_f1 = np.std(filtered_f1)

print(f"Precision of all regions: {mean_region_precision}±{std_region_precision}")
print(f"Recall of all regions: {mean_region_recall}±{std_region_recall}")
print(f"F1 score of all regions: {mean_region_f1}±{std_region_f1}")
print("\n")

# 打印出现错误的 accnum
for region, accnums in region_FP_accnums.items():
    print(f"Region: {region}")
    print(f"False positive accnums: {accnums}")
    print("\n")
    
for region, accnums in region_FN_accnums.items():
    print(f"Region: {region}")
    print(f"False negative accnums: {accnums}")
    print("\n")
    
# 把之前打印出来的结果保存到text文件中
save_path = os.path.dirname(results_path)

with open(os.path.join(save_path, 'region_classification_report.txt'), 'w') as file:
    file.write("Accuracy for each sample:\n")
    file.write(f"Mean accuracy: {mean_acc}\n")
    file.write(f"Standard deviation: {std_acc}\n")
    file.write("\n")
    
    file.write("Precision, recall and F1 score for each region:\n")
    for region in REGIONS:
        file.write(f"{region}:\n")
        file.write(f"Precision: {region_precision[region]}\n")
        file.write(f"Recall: {region_recall[region]}\n")
        file.write(f"F1 score: {region_f1[region]}\n")
        file.write("\n")
        
    file.write(f"Precision of all regions: {mean_region_precision}±{std_region_precision}\n")
    file.write(f"Recall of all regions: {mean_region_recall}±{std_region_recall}\n")
    file.write(f"F1 score of all regions: {mean_region_f1}±{std_region_f1}\n")
    file.write("\n")
    
    file.write("FP and FN accnums for each region:\n")
    for region, accnums in region_FP_accnums.items():
        file.write(f"Region: {region}\n")
        file.write(f"False positive accnums: {accnums}\n")
        file.write("\n")
        
    for region, accnums in region_FN_accnums.items():
        file.write(f"Region: {region}\n")
        file.write(f"False negative accnums: {accnums}\n")
        file.write("\n")
    

