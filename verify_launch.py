"""Pre-launch verification: tests EVERY disease target across ALL datasets"""
import os, sys
sys.path.insert(0, r'l:\MTP')
sys.path.insert(0, r'l:\MTP\src\data')
sys.path.insert(0, r'l:\MTP\src')
from pathlib import Path
from dataset import UniversalMedicalDataset

TARGET_DISEASES = {
    "NIH": [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ],
    "MIMIC": [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged_Cardiomediastinum",
        "Fracture", "Lung_Lesion", "Lung_Opacity", "No_Finding", "Pleural_Effusion",
        "Pleural_Other", "Pneumonia", "Pneumothorax", "Support_Devices"
    ],
    "ISIC_2024": ["Melanoma"],
    "ISIC_2018": ["Melanoma"],
    "covid-19": ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"],
    "rsna-pneumonia-detection-challenge": ["Pneumonia"],
    "HAM10000": ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"],
    "ODIR": ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"],
    "diabetic-retinopathy-detection": ["Diabetic_Retinopathy"],
    "OASIS": ["Dementia"]
}

DATASET_PATHS = {
    "NIH": r"l:\MTP\datasets\chestxray\NIH",
    "MIMIC": r"l:\MTP\datasets\chestxray\MIMIC",
    "ISIC_2024": r"l:\MTP\datasets\dermatology\ISIC_2024",
    "ISIC_2018": r"l:\MTP\datasets\dermatology\ISIC_2018",
    "covid-19": r"l:\MTP\datasets\chestxray\covid-19",
    "rsna-pneumonia-detection-challenge": r"l:\MTP\datasets\chestxray\rsna-pneumonia-detection-challenge",
    "HAM10000": r"l:\MTP\datasets\dermatology\HAM10000",
    "ODIR": r"l:\MTP\datasets\ophthalmology\ODIR",
    "diabetic-retinopathy-detection": r"l:\MTP\datasets\ophthalmology\diabetic-retinopathy-detection",
    "OASIS": r"l:\MTP\datasets\brain\OASIS",
}

print(f"{'#':>3s}  {'Dataset':30s}  {'Disease':28s}  {'Samples':>8s}  {'Img Shape':>15s}  {'Meta':>6s}  {'Status'}")
print("-" * 115)

num = 0
ok = 0
fail = 0
warn = 0

for ds_name, diseases in TARGET_DISEASES.items():
    path = DATASET_PATHS[ds_name]
    for disease in diseases:
        num += 1
        try:
            dataset = UniversalMedicalDataset(domain_path=path, split='train', target_disease=disease)
            if len(dataset) == 0:
                print(f"{num:3d}  {ds_name:30s}  {disease:28s}  {'0':>8s}  {'N/A':>15s}  {'N/A':>6s}  WARN: 0 samples")
                warn += 1
            else:
                img, meta, label = dataset[0]
                shape = f"{list(img.shape)}"
                meta_d = f"[{meta.shape[0]}]"
                print(f"{num:3d}  {ds_name:30s}  {disease:28s}  {len(dataset):8d}  {shape:>15s}  {meta_d:>6s}  OK")
                ok += 1
        except Exception as e:
            err = str(e)[:60]
            print(f"{num:3d}  {ds_name:30s}  {disease:28s}  {'ERR':>8s}  {'N/A':>15s}  {'N/A':>6s}  FAIL: {err}")
            fail += 1

print("-" * 115)
print(f"TOTAL: {num} disease targets | {ok} OK | {warn} WARN (0 samples) | {fail} FAIL")
if fail == 0 and warn == 0:
    print("ALL DISEASE TARGETS READY!")
elif fail == 0:
    print(f"{ok} ready to train, {warn} will be auto-skipped (0 positive samples)")
