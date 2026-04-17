import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    np.random.seed(seed)

def main():
    set_seed(42)
    labels_path = 'dataset/raw/labels.csv'
    output_path = 'dataset/splits/patient_split.json'
    
    if not os.path.exists(labels_path):
        print(f"Warning: {labels_path} not found. Ensure the dataset metadata is placed here before running.")
        return

    df = pd.read_csv(labels_path)
    
    # Identify Patient ID column (usually 'Patient ID' in NIH dataset)
    patient_col = 'Patient ID'
    if patient_col not in df.columns:
        if 'patient_id' in df.columns.str.lower():
            patient_col = df.columns[df.columns.str.lower() == 'patient_id'][0]
        else:
            raise KeyError("Could not find Patient ID column in labels.csv")

    # Get unique patients
    unique_patients = df[patient_col].unique()
    print(f"Total unique patients: {len(unique_patients)}")

    # Patient-wise split: 70% train, 30% temp
    train_patients, temp_patients = train_test_split(unique_patients, test_size=0.30, random_state=42)
    # Split temp into 15% val, 15% test (which is 50% of the 30%)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)

    # Convert to native Python types for JSON serialization
    train_patients = [int(p) if isinstance(p, np.integer) else p for p in train_patients]
    val_patients = [int(p) if isinstance(p, np.integer) else p for p in val_patients]
    test_patients = [int(p) if isinstance(p, np.integer) else p for p in test_patients]

    splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Patient-wise split complete. Saved to {output_path}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

if __name__ == "__main__":
    main()
