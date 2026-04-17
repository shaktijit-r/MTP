import os
import pandas as pd
from pathlib import Path

def parse_metadata(domain_name, dataset_name, labels_dir, target_disease):
    """
    Reads the dataset-specific CSV/Excel metadata and returns a normalized dictionary.
    New Multi-Modal Output: { "image_filename.ext": (Label_Float, [Normalized_Age, Is_Male, Is_Female, Has_Meta]) }
    """
    labels_dir = Path(labels_dir)
    labels = {}
    
    csv_path = labels_dir / "labels.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return labels

    def extract_demographics(row, age_col, sex_col):
        """Helper to universally extract and normalize Age, Sex and Advanced Tumoral Metrics into a dynamic N-D Tensor."""
        
        # Max Efficiency Optimization: Pure Vision Datasets drop the EHR Tensor entirely
        if dataset_name in ["COVID-19 Radiography", "covid-19", "Diabetic Retinopathy", "diabetic-retinopathy-detection"]:
            return []
            
        age_val = row.get(age_col, None)
        sex_val = row.get(sex_col, None)
        
        has_meta = 1.0
        
        # Parse Age
        try:
            if pd.isna(age_val): raise ValueError
            # Z-Score Standard Scaling (Mean=55.0, Std=20.0)
            age_norm = (float(age_val) - 55.0) / 20.0
            age_norm = min(max(age_norm, -3.0), 3.0) # Clamp -3 to 3 STD
        except:
            age_norm = 0.0
            has_meta = -1.0
            
        # Parse Sex
        is_m, is_f = -1.0, -1.0
        try:
            if pd.isna(sex_val): raise ValueError
            s = str(sex_val).strip().lower()
            if s in ['m', 'male']:
                is_m = 1.0
                is_f = -1.0
            elif s in ['f', 'female']:
                is_m = -1.0
                is_f = 1.0
            else:
                is_m, is_f = 0.0, 0.0
                has_meta = -1.0
        except:
            is_m, is_f = 0.0, 0.0
            has_meta = -1.0
            
        base_meta = [age_norm, is_m, is_f, has_meta]
        
        # Basic Datasets (Chest X-Rays) only require Age/Sex (4-D max efficiency)
        if dataset_name in ["NIH", "MIMIC", "RSNA Pneumonia", "rsna-pneumonia-detection-challenge"]:
            return base_meta
        
        # OASIS: Enriched Brain Demographics (11-D tensor)
        if dataset_name == "OASIS":
            # Hand (one-hot: R=1, L=0)
            hand_val = row.get('Hand', None)
            try:
                if pd.isna(hand_val): raise ValueError
                is_right = 1.0 if str(hand_val).strip().upper() == 'R' else -1.0
            except:
                is_right = 0.0
            
            # Education (1-5, normalized to 0-1)
            try:
                educ = float(row.get('Educ', None))
                if pd.isna(educ): raise ValueError
                educ_norm = educ / 5.0
            except:
                educ_norm = 0.0
            
            # SES (1-5, normalized to 0-1)
            try:
                ses = float(row.get('SES', None))
                if pd.isna(ses): raise ValueError
                ses_norm = ses / 5.0
            except:
                ses_norm = 0.0
            
            # MMSE (0-30, normalized to 0-1)
            try:
                mmse = float(row.get('MMSE', None))
                if pd.isna(mmse): raise ValueError
                mmse_norm = mmse / 30.0
            except:
                mmse_norm = 0.0
            
            # eTIV (z-score: mean=1482, std=176)
            try:
                etiv = float(row.get('eTIV', None))
                if pd.isna(etiv): raise ValueError
                etiv_norm = (etiv - 1482.0) / 176.0
                etiv_norm = min(max(etiv_norm, -3.0), 3.0)
            except:
                etiv_norm = 0.0
            
            # nWBV (already 0-1 range, use directly)
            try:
                nwbv = float(row.get('nWBV', None))
                if pd.isna(nwbv): raise ValueError
            except:
                nwbv = 0.0
            
            # ASF (z-score: mean=1.2, std=0.13)
            try:
                asf = float(row.get('ASF', None))
                if pd.isna(asf): raise ValueError
                asf_norm = (asf - 1.2) / 0.13
                asf_norm = min(max(asf_norm, -3.0), 3.0)
            except:
                asf_norm = 0.0
            
            return base_meta + [is_right, educ_norm, ses_norm, mmse_norm, etiv_norm, nwbv, asf_norm]
            
        # Advanced Localization (One-Hot 9)
        loc = str(row.get('Localization', '')).strip().lower()
        locs = ['head/neck', 'trunk', 'upper extremity', 'lower extremity', 'back', 'abdomen', 'chest', 'face']
        loc_vec = [1.0 if l in loc else 0.0 for l in locs]
        is_other_loc = 1.0 if loc and sum(loc_vec) == 0 else 0.0
        loc_vec.append(is_other_loc)

        # Advanced Tumoral Physics (7)
        try: size_mm = float(row.get('Tumor_Size_MM', 0.0)) / 100.0  # normalize assuming max 10cm
        except: size_mm = 0.0
        
        tbp_keys = ['TBP_Color_L', 'TBP_Color_A', 'TBP_Color_B', 'TBP_Color_C', 'TBP_Color_Std_Mean', 'TBP_Eccentricity']
        tbp_vec = []
        for k in tbp_keys:
            try: tbp_vec.append(float(row.get(k, 0.0)))
            except: tbp_vec.append(0.0)

        # Metadata Integrity Flag
        has_adv_ehr = 1.0 if ('Localization' in row) or ('Tumor_Size_MM' in row) else 0.0

        # Full 21-D Advanced Geometry Matrix for Dermatology
        return base_meta + loc_vec + [size_mm] + tbp_vec + [has_adv_ehr]

    if dataset_name == "NIH":
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_name = row['Image Index']
            findings = str(row['Finding Labels'])
            label = 0.95 if target_disease in findings else 0.05
            meta = extract_demographics(row, 'Patient Age', 'Patient Gender')
            labels[img_name] = (label, meta)

    elif dataset_name == "MIMIC":
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_name = row['Image Index']
            if target_disease in df.columns:
                label = 0.95 if row[target_disease] == 1 else 0.05
            else:
                label = 0.05
            meta = extract_demographics(row, 'Age', 'Sex')  # Pure vision fallback, returns [4]
            labels[img_name] = (label, meta)

    elif dataset_name in ["ISIC_2024", "ISIC_2018", "ISIC"]:
        df = pd.read_csv(csv_path, low_memory=False)
        for _, row in df.iterrows():
            isic_id = row['isic_id']
            if 'diagnosis_1' in row and not pd.isna(row['diagnosis_1']):
                label = 0.95 if str(row['diagnosis_1']).lower() == 'malignant' else 0.05
            elif 'target' in row and not pd.isna(row['target']):
                label = 0.95 if float(row['target']) == 1.0 else 0.05
            elif 'benign_malignant' in row and not pd.isna(row['benign_malignant']):
                label = 0.95 if str(row['benign_malignant']).lower() == 'malignant' else 0.05
            else:
                label = 0.05
                
            meta = extract_demographics(row, 'age_approx', 'sex')
            labels[f"{isic_id}.jpg"] = (label, meta)

    elif dataset_name == "ODIR":
        df = pd.read_csv(csv_path)
        col_map = {
            "Normal": "N", "Diabetes": "D", "Glaucoma": "G", "Cataract": "C", 
            "AMD": "A", "Hypertension": "H", "Myopia": "M", "Other": "O"
        }
        target_col = col_map.get(target_disease, target_disease)
        for _, row in df.iterrows():
            filename = row.get('filename', "")
            if 'filename' in row and target_col in df.columns:
                val = row[target_col]
                label = 0.95 if val == 1 else 0.05
                meta = extract_demographics(row, 'Patient Age', 'Patient Sex')
                labels[filename] = (label, meta)

    elif dataset_name == "rsna-pneumonia-detection-challenge":
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_name = row['Image Index']
            if target_disease in df.columns:
                label = 0.95 if row[target_disease] == 1.0 else 0.05
            else:
                label = 0.05
            meta = extract_demographics(row, 'Patient Age', 'Patient Sex')
            labels[img_name] = (label, meta)

    elif dataset_name in ["HAM10000", "ham10000_integrated"]:
        df = pd.read_csv(csv_path)
        # One-hot columns: bkl, nv, df, mel, vasc, bcc, akiec
        for _, row in df.iterrows():
            img_name = row['Image Index']
            if target_disease in df.columns:
                label = 0.95 if row[target_disease] == 1.0 else 0.05
            else:
                label = 0.05
            meta = extract_demographics(row, 'Patient Age', 'Patient Sex')
            labels[img_name] = (label, meta)

    elif dataset_name == "covid-19":
        df = pd.read_csv(csv_path)
        # One-hot columns: COVID, Lung_Opacity, Normal, Viral_Pneumonia
        for _, row in df.iterrows():
            img_name = row['Image Index']
            if target_disease in df.columns:
                label = 0.95 if row[target_disease] == 1.0 else 0.05
            else:
                label = 0.05
            meta = extract_demographics(row, 'Age', 'Sex')  # Pure vision, returns []
            labels[img_name] = (label, meta)

    elif dataset_name == "diabetic-retinopathy-detection":
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_name = row['Image Index']
            if target_disease in df.columns:
                label = 0.95 if row[target_disease] == 1 else 0.05
            else:
                label = 0.05
            meta = extract_demographics(row, 'Age', 'Sex')  # Pure vision, returns []
            labels[img_name] = (label, meta)

    elif dataset_name == "OASIS":
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_name = row['Image Index']
            cdr = row.get('CDR', 0.0)
            label = 0.95 if not pd.isna(cdr) and cdr > 0.0 else 0.05
            meta = extract_demographics(row, 'Age', 'M/F')
            labels[img_name] = (label, meta)
            
    return labels
