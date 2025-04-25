import pandas as pd
import re
import os

results_path = "/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports.csv"
df = pd.read_csv(results_path)

# 读取GT_combined_report和Pred_combined_report列
gts = df['GT_combined_report']
preds = df['Pred_combined_report']

filtered_gts = []
filtered_preds = []

# 处理每一个sample
for gt, pred in zip(gts, preds):
    # 使用正则表达式移除每个区域描述的前缀
    gt = re.sub(r"The region \d+ is [^:]+: ", "", gt)
    pred = re.sub(r"The region \d+ is [^:]+: ", "", pred)
    # gt = re.sub(r"The region \d:\s*", "", gt)
    # pred = re.sub(r"The region \d:\s*", "", pred)

    # 替换多余的空格为一个空格，并去除可能出现在句子前后的空格
    gt = re.sub(r'\s+', ' ', gt).strip()
    pred = re.sub(r'\s+', ' ', pred).strip()

    filtered_gts.append(gt)
    filtered_preds.append(pred)

# 将处理后的GT_combined_report和Pred_combined_report列替换原来的列
df['GT_combined_report'] = filtered_gts
df['Pred_combined_report'] = filtered_preds

# 保存处理后的csv文件
save_dir = os.path.dirname(results_path)
save_name = os.path.basename(results_path)
# 去掉文件名中的后缀
save_name = save_name.split(".")[0]
df.to_csv(os.path.join(save_dir, save_name + "_rm_region_text.csv"), index=False)

