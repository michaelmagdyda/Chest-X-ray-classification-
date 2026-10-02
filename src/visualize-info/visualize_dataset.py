import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

# 1. Parse command line arguments
parser = argparse.ArgumentParser(description="Visualize NIH Chest X-ray Dataset")
parser.add_argument("--csv_path", type=str, default='Data_Entry_2017.csv', help="Path to Data_Entry_2017.csv")
parser.add_argument("--output_dir", type=str, default='info', help="Directory to save charts")
args = parser.parse_args()

csv_path = args.csv_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("Loading dataset metadata...")
df = pd.read_csv(csv_path)

# Disease categories (excluding No Finding for some counts)
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Set matplotlib style for premium visual look
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#1e1e1e'
plt.rcParams['text.color'] = '#e0e0e0'
plt.rcParams['axes.labelcolor'] = '#e0e0e0'
plt.rcParams['xtick.color'] = '#a0a0a0'
plt.rcParams['ytick.color'] = '#a0a0a0'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['font.size'] = 10

# ----------------------------------------------------
# 1. CLASS DISTRIBUTION CHART
# ----------------------------------------------------
print("Generating class distribution chart...")
label_counts = {}
single_label_counts = {}

# Initialize counts
for d in DISEASES + ['No Finding']:
    label_counts[d] = 0
    single_label_counts[d] = 0

for labels in df['Finding Labels']:
    lbl_list = labels.split('|')
    for l in lbl_list:
        if l in label_counts:
            label_counts[l] += 1
    if len(lbl_list) == 1:
        single_label_counts[lbl_list[0]] += 1

# Sort by count
sorted_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)
all_counts = [label_counts[l] for l in sorted_labels]
single_counts = [single_label_counts[l] for l in sorted_labels]

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(sorted_labels))

rects1 = ax.bar(index - bar_width/2, all_counts, bar_width, label='Total Occurrences', color='#00adb5')
rects2 = ax.bar(index + bar_width/2, single_counts, bar_width, label='Single Label Only', color='#393e46')

ax.set_xlabel('Thorax Pathologies / Findings')
ax.set_ylabel('Number of Images')
ax.set_title('Pathology Distribution in NIH Chest X-ray Dataset', fontsize=14, color='#ffffff', pad=15)
ax.set_xticks(index)
ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
ax.legend(facecolor='#1e1e1e', edgecolor='#333333')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=200, facecolor='#121212')
plt.close()

# ----------------------------------------------------
# 2. CO-OCCURRENCE MATRIX HEATMAP
# ----------------------------------------------------
print("Generating disease co-occurrence heatmap...")
co_matrix = pd.DataFrame(0, index=DISEASES, columns=DISEASES)

for labels in df['Finding Labels']:
    lbl_list = labels.split('|')
    valid_diseases = [l for l in lbl_list if l in DISEASES]
    for d1 in valid_diseases:
        for d2 in valid_diseases:
            co_matrix.loc[d1, d2] += 1

# Normalize diagonal or keep absolute? Let's display log scale for readability since counts vary widely
co_matrix_log = np.log1p(co_matrix.values)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(co_matrix_log, cmap='viridis')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Log(Co-occurrences + 1)", rotation=-90, va="bottom")

# Set tick labels
ax.set_xticks(np.arange(len(DISEASES)))
ax.set_yticks(np.arange(len(DISEASES)))
ax.set_xticklabels(DISEASES, rotation=45, ha='right')
ax.set_yticklabels(DISEASES)

# Annotate with actual count integers
for i in range(len(DISEASES)):
    for j in range(len(DISEASES)):
        count = co_matrix.iloc[i, j]
        # Choose text color based on matrix value for readability
        val = co_matrix_log[i, j]
        color = "white" if val < 7.0 else "black"
        ax.text(j, i, f"{count}", ha="center", va="center", color=color, fontsize=8)

ax.set_title("Thorax Disease Co-occurrence Matrix (Absolute Counts)", fontsize=14, color='#ffffff', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "disease_cooccurrence.png"), dpi=200, facecolor='#121212')
plt.close()

# ----------------------------------------------------
# 3. PATIENT DEMOGRAPHICS (AGE & GENDER)
# ----------------------------------------------------
print("Generating patient demographics charts...")
# For patient stats, drop duplicate patients to avoid bias from multiple scans
unique_patients_df = df.drop_duplicates(subset=['Patient ID'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# A. Age Distribution
# Filter out potential age entry typos (e.g. age > 120)
valid_ages = unique_patients_df[unique_patients_df['Patient Age'] <= 100]['Patient Age']
ax1.hist(valid_ages, bins=20, color='#00adb5', edgecolor='#121212', alpha=0.8)
ax1.set_title('Patient Age Distribution (Unique Patients)', fontsize=12, color='#ffffff', pad=10)
ax1.set_xlabel('Age')
ax1.set_ylabel('Number of Patients')
ax1.grid(True, linestyle='--', alpha=0.3)

# B. Gender Distribution
gender_counts = unique_patients_df['Patient Gender'].value_counts()
labels = [f"Male ({gender_counts.get('M', 0)})", f"Female ({gender_counts.get('F', 0)})"]
colors = ['#393e46', '#00adb5']
explode = (0.05, 0)

ax2.pie(gender_counts, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=140,
        textprops={'color': '#ffffff'})
ax2.set_title('Patient Gender Split (Unique Patients)', fontsize=12, color='#ffffff', pad=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "patient_demographics.png"), dpi=200, facecolor='#121212')
plt.close()

# ----------------------------------------------------
# 4. GENERATING MARKDOWN SUMMARY REPORT
# ----------------------------------------------------
print("Writing dataset summary report...")
total_images = len(df)
total_patients = df['Patient ID'].nunique()
avg_scans_per_patient = total_images / total_patients

report_content = f"""# NIH Chest X-ray Dataset Analysis Report

This folder contains the visual analysis and characteristics of the NIH ChestX-ray14 dataset.

---

## 1. Core Summary Metrics
* **Total X-ray Images**: {total_images:,}
* **Unique Patients**: {total_patients:,}
* **Average Scans per Patient**: {avg_scans_per_patient:.2f}
* **No Finding Ratio**: {label_counts['No Finding'] * 100 / total_images:.2f}% ({label_counts['No Finding']:,} images)
* **Single Pathology Ratio**: {sum(single_label_counts[d] for d in DISEASES) * 100 / total_images:.2f}% ({sum(single_label_counts[d] for d in DISEASES):,} images)
* **Multi-label Pathology Ratio**: {(total_images - label_counts['No Finding'] - sum(single_label_counts[d] for d in DISEASES)) * 100 / total_images:.2f}% ({(total_images - label_counts['No Finding'] - sum(single_label_counts[d] for d in DISEASES)):,} images)

---

## 2. Visualizations

### A. Class Distribution
This chart shows the absolute occurrences of each pathology in the dataset, split by whether they occur alone (Single Label) or with other co-occurring diseases.

![Class Distribution](class_distribution.png)

### B. Disease Co-occurrence Matrix
A heatmap displaying the co-occurrence frequencies of the 14 pathologies. It highlights which conditions commonly exist together (such as **Infiltration** and **Atelectasis**).

![Disease Co-occurrence Heatmap](disease_cooccurrence.png)

### C. Patient Demographics
Histograms representing unique patient age distribution (filtered for normal biological range <= 100 years) and the gender split.

![Patient Demographics](patient_demographics.png)

---

*This report was generated automatically on 2026-07-18.*
"""

with open(os.path.join(output_dir, "dataset_summary.md"), 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"All outputs successfully generated and saved in: {output_dir}")
