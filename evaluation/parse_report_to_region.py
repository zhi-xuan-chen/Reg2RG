import re
import pandas as pd
import os

# Define the anatomical regions
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

def parse_report_by_name(report_text):
    """
    Parses the given report text into a dictionary based on anatomical regions.
    
    Parameters:
        report_text (str): The complete report text to be parsed.
    
    Returns:
        dict: A dictionary where keys are region names and values are corresponding reports as strings.
              Empty entries (including those with only whitespace) are removed.
    """
    region_reports = {}

    # Use regex to extract region-specific reports
    for region_name in regions:
        pattern = rf"The region \d+ is {region_name}:(.+?)(?=(The region \d+ is|$))"
        matches = re.findall(pattern, report_text, re.DOTALL)

        # Clean and concatenate reports into a single string
        cleaned_report = " ".join(match[0].strip() for match in matches if match[0].strip())

        # Only add non-empty reports
        if cleaned_report:
            region_reports[region_name] = cleaned_report

    return region_reports


# Input CSV file containing combined reports
combined_report_file = '/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports.csv'
results_dir = os.path.dirname(combined_report_file)

# Read input CSV file
combined_reports = pd.read_csv(combined_report_file)

# Dictionary to store region-specific data
region_data = {region: [] for region in regions}

# Process each row in the dataset
for _, row in combined_reports.iterrows():
    accnum = row['AccNum']
    pred_report = row['Pred_combined_report']
    gt_report = row['GT_combined_report']
    
    # Parse region-specific reports
    pred_region_reports = parse_report_by_name(pred_report)
    gt_region_reports = parse_report_by_name(gt_report)

    # Find common regions that exist in both GT and Pred
    common_regions = set(pred_region_reports.keys()).intersection(gt_region_reports.keys())

    # Store data only for common regions
    for region in common_regions:
        region_data[region].append({
            'AccNum': accnum,
            'GT_combined_report': gt_region_reports[region],
            'Pred_combined_report': pred_region_reports[region]
        })

# Save each region's data into separate CSV files
for region, data in region_data.items():
    if data:  # Ensure we don't create empty CSV files
        region_df = pd.DataFrame(data)
        region_csv_path = os.path.join(results_dir, f"{region}_reports.csv")
        region_df.to_csv(region_csv_path, index=False)

print(f"Region-specific reports have been saved in {results_dir}")