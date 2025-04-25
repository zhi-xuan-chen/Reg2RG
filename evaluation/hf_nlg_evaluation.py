import evaluate
import pandas as pd
import numpy as np
import torch
import random
import os
import json
from pycocoevalcap.cider.cider import Cider

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)  # 例如使用42作为随机种子

# 创建评估器实例
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

results_path = "/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports_rm_region_text.csv"

gt_report_column = 'GT_combined_report'
pred_report_column = 'Pred_combined_report'

df = pd.read_csv(results_path)

# print number of samples
print(f"Number of samples: {len(df)}")

# references
gts = df[gt_report_column].tolist()
# convert each reference to a list
references = [[gt] for gt in gts]

# predictions
predictions = df[pred_report_column].tolist()

# 计算 BLEU 得分及其标准差
bleu_scores = {n: [] for n in range(1, 5)}
for prediction, reference in zip(predictions, references):
    for n in range(1, 5):
        result = bleu.compute(predictions=[prediction], references=[reference], max_order=n, smooth=True)
        bleu_scores[n].append(result['bleu'])

for n in range(1, 5):
    mean_bleu = np.mean(bleu_scores[n])
    std_bleu = np.std(bleu_scores[n])
    print(f"BLEU-{n}: mean = {mean_bleu}, std = {std_bleu}")

# 计算 METEOR 得分及其标准差
meteor_scores = []
for prediction, reference in zip(predictions, references):
    result = meteor.compute(predictions=[prediction], references=[reference])
    meteor_scores.append(result['meteor'])

mean_meteor = np.mean(meteor_scores)
std_meteor = np.std(meteor_scores)
print(f"METEOR: mean = {mean_meteor}, std = {std_meteor}")

# 计算 ROUGE-1, ROUGE-2, 和 ROUGE-L 得分及其标准差
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
for prediction, reference in zip(predictions, references):
    result = rouge.compute(predictions=[prediction], references=[reference])
    rouge1_scores.append(result['rouge1'])
    rouge2_scores.append(result['rouge2'])
    rougeL_scores.append(result['rougeL'])

mean_rouge1 = np.mean(rouge1_scores)
std_rouge1 = np.std(rouge1_scores)
print(f"ROUGE-1: mean = {mean_rouge1}, std = {std_rouge1}")

mean_rouge2 = np.mean(rouge2_scores)
std_rouge2 = np.std(rouge2_scores)
print(f"ROUGE-2: mean = {mean_rouge2}, std = {std_rouge2}")

mean_rougeL = np.mean(rougeL_scores)
std_rougeL = np.std(rougeL_scores)
print(f"ROUGE-L: mean = {mean_rougeL}, std = {std_rougeL}")

# 初始化 CIDEr 计算器
cider_scorer = Cider()

gts = {str(idx): reference for idx, reference in enumerate(references)}
res = {str(idx): [prediction] for idx, prediction in enumerate(predictions)}

# 计算 CIDEr 得分
cider_score, cider_scores = cider_scorer.compute_score(gts, res)
mean_cider = np.mean(cider_scores)
std_cider = np.std(cider_scores)
print(f"CIDEr: mean = {mean_cider}, std = {std_cider}")

# 比较整体计算结果
# 计算不同 n-gram 长度的 BLEU 得分
for n in range(1, 5):
    overall_result = bleu.compute(predictions=predictions, references=references, max_order=n, smooth=True)
    print(f"Overall BLEU-{n}:", overall_result['bleu'])

# 计算 METEOR 得分
overall_meteor_result = meteor.compute(predictions=predictions, references=references)
print("Overall METEOR:", overall_meteor_result['meteor'])

# 计算 ROUGE 得分
overall_rouge_result = rouge.compute(predictions=predictions, references=references)
print("Overall ROUGE-1:", overall_rouge_result['rouge1'])
print("Overall ROUGE-2:", overall_rouge_result['rouge2'])
print("Overall ROUGE-L:", overall_rouge_result['rougeL'])

# 保存评估结果到 csv, 有标准差的要以 mean±std 的形式保存

# 保存 BLEU-1, BLEU-2, BLEU-3, BLEU-4 的平均得分和标准差
results = {}
for n in range(1, 5):
    mean_bleu = np.mean(bleu_scores[n])
    std_bleu = np.std(bleu_scores[n])
    results[f"BLEU-{n}"] = f"{mean_bleu:.4f}±{std_bleu:.4f}"

# 保存 METEOR 的平均得分和标准差
results["METEOR"] = f"{mean_meteor:.4f}±{std_meteor:.4f}"

# 保存 ROUGE-1, ROUGE-2, ROUGE-L 的平均得分和标准差
results["ROUGE-1"] = f"{mean_rouge1:.4f}±{std_rouge1:.4f}"
results["ROUGE-2"] = f"{mean_rouge2:.4f}±{std_rouge2:.4f}"
results["ROUGE-L"] = f"{mean_rougeL:.4f}±{std_rougeL:.4f}"
results["CIDEr"] = f"{mean_cider:.4f}±{std_cider:.4f}"

# 保存整体计算结果

# 保存不同 n-gram 长度的 BLEU 得分
for n in range(1, 5):
    overall_result = bleu.compute(predictions=predictions, references=references, max_order=n, smooth=True)
    results[f"Overall_BLEU-{n}"] = f"{overall_result['bleu']:.4f}"

# 保存 METEOR 得分
overall_meteor_result = meteor.compute(predictions=predictions, references=references)
results["Overall_METEOR"] = f"{overall_meteor_result['meteor']:.4f}"

# 保存 ROUGE 得分
overall_rouge_result = rouge.compute(predictions=predictions, references=references)
results["Overall_ROUGE-1"] = f"{overall_rouge_result['rouge1']:.4f}"
results["Overall_ROUGE-2"] = f"{overall_rouge_result['rouge2']:.4f}"
results["Overall_ROUGE-L"] = f"{overall_rouge_result['rougeL']:.4f}"

# 保存到 csv
results_df = pd.DataFrame(results, index=[0])
save_dir = os.path.dirname(results_path)
results_df.to_csv(os.path.join(save_dir, "nlg_evaluation_results.csv"), index=False)


