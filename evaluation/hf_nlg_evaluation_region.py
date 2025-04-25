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

results_dir = "/home/chenzhixuan/Workspace/FineGrainedCTRG/results/regionRG-radfm_radgenome_mask-crop-regions-image_remain_wrong"

gt_report_column = 'GT_combined_report'
pred_report_column = 'Pred_combined_report'

all_results = []

for region in regions:
    file_path = os.path.join(results_dir, f"{region}_reports.csv")
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping...")
        continue

    df = pd.read_csv(file_path)

    # print number of samples
    print(f"Number of samples in {region}: {len(df)}")

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

    bleu_mean_std = {}
    for n in range(1, 5):
        mean_bleu = np.mean(bleu_scores[n])
        std_bleu = np.std(bleu_scores[n])
        bleu_mean_std[f"BLEU-{n}"] = f"{mean_bleu:.4f}±{std_bleu:.4f}"

    # 计算 METEOR 得分及其标准差
    meteor_scores = []
    for prediction, reference in zip(predictions, references):
        result = meteor.compute(predictions=[prediction], references=[reference])
        meteor_scores.append(result['meteor'])

    mean_meteor = np.mean(meteor_scores)
    std_meteor = np.std(meteor_scores)
    meteor_mean_std = f"{mean_meteor:.4f}±{std_meteor:.4f}"

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
    rouge1_mean_std = f"{mean_rouge1:.4f}±{std_rouge1:.4f}"

    mean_rouge2 = np.mean(rouge2_scores)
    std_rouge2 = np.std(rouge2_scores)
    rouge2_mean_std = f"{mean_rouge2:.4f}±{std_rouge2:.4f}"

    mean_rougeL = np.mean(rougeL_scores)
    std_rougeL = np.std(rougeL_scores)
    rougeL_mean_std = f"{mean_rougeL:.4f}±{std_rougeL:.4f}"

    # 初始化 CIDEr 计算器
    cider_scorer = Cider()

    gts_dict = {str(idx): reference for idx, reference in enumerate(references)}
    res_dict = {str(idx): [prediction] for idx, prediction in enumerate(predictions)}

    # 计算 CIDEr 得分
    cider_score, cider_scores = cider_scorer.compute_score(gts_dict, res_dict)
    mean_cider = np.mean(cider_scores)
    std_cider = np.std(cider_scores)
    cider_mean_std = f"{mean_cider:.4f}±{std_cider:.4f}"

    # 比较整体计算结果
    # 计算不同 n-gram 长度的 BLEU 得分
    overall_bleu = {}
    for n in range(1, 5):
        overall_result = bleu.compute(predictions=predictions, references=references, max_order=n, smooth=True)
        overall_bleu[f"Overall_BLEU-{n}"] = f"{overall_result['bleu']:.4f}"

    # 计算 METEOR 得分
    overall_meteor_result = meteor.compute(predictions=predictions, references=references)
    overall_meteor = f"{overall_meteor_result['meteor']:.4f}"

    # 计算 ROUGE 得分
    overall_rouge_result = rouge.compute(predictions=predictions, references=references)
    overall_rouge1 = f"{overall_rouge_result['rouge1']:.4f}"
    overall_rouge2 = f"{overall_rouge_result['rouge2']:.4f}"
    overall_rougeL = f"{overall_rouge_result['rougeL']:.4f}"

    # 整理结果
    result = {
        "region": region,
        **bleu_mean_std,
        "METEOR": meteor_mean_std,
        "ROUGE-1": rouge1_mean_std,
        "ROUGE-2": rouge2_mean_std,
        "ROUGE-L": rougeL_mean_std,
        "CIDEr": cider_mean_std,
        **overall_bleu,
        "Overall_METEOR": overall_meteor,
        "Overall_ROUGE-1": overall_rouge1,
        "Overall_ROUGE-2": overall_rouge2,
        "Overall_ROUGE-L": overall_rougeL
    }
    all_results.append(result)

# 保存到 csv
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(results_dir, "nlg_evaluation_results_all_regions.csv"), index=False)