import os
from pathlib import Path
import subprocess
import sys

# Exhaustive Combinatorial Target Grid for all 5 Supported Modalities
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
    "OASIS": ["Dementia"]
}

def main():
    base_dir = Path(r"l:\MTP\datasets")
    if not base_dir.exists():
        print(f"Error: Base dataset directory {base_dir} does not exist.")
        sys.exit(1)

    domains = ["chestxray", "dermatology", "ophthalmology", "brain"]
    
    found_datasets = []
    for domain in domains:
        domain_path = base_dir / domain
        if not domain_path.exists():
            continue
            
        for dataset_name in os.listdir(domain_path):
            dataset_path = domain_path / dataset_name
            if not dataset_path.is_dir():
                continue
                
            images_dir = dataset_path / "images"
            if images_dir.exists():
                count = len(os.listdir(images_dir))
                if count > 0:
                    found_datasets.append((dataset_name, dataset_path, count))
                    
    if not found_datasets:
        print("No datasets with populated `images/` folders found. Aborting.")
        sys.exit(0)
        
    print(f"Found {len(found_datasets)} populated datasets active for training:")
    for name, path, count in found_datasets:
        print(f"  - {name} ({count} images)")
        
    print("\nStarting multi-domain batch orchestration...\n")
    
    # We use the python executable that's running this script to launch children
    # ensuring the venv stays consistent
    python_exe = sys.executable 
    
    for dataset_name, dataset_path, count in found_datasets:
        # Fetch the entire array of target pathologies for this specific dataset
        targets = TARGET_DISEASES.get(dataset_name, ["Pathology"])
        
        for target_disease in targets:
            print(f"{'='*50}")
            print(f"INITIATING TRAINING: {dataset_name} | Target: {target_disease}")
            print(f"{'='*50}")
        
            # Build the command
            train_script = str(Path(r"l:\MTP\src\models\train_baseline.py"))
            
            cmd = [
                python_exe, 
                train_script, 
                "--domain_path", str(dataset_path),
                "--target_disease", target_disease
            ]
            
            # Set explicitly to MTP project root for imports
            env = os.environ.copy()
            env["PYTHONPATH"] = r"l:\MTP;l:\MTP\src\data;l:\MTP\src"
            
            try:
                subprocess.run(cmd, env=env, cwd=r"l:\MTP", check=True)
                print(f"\nSUCCESS: Training completed for {dataset_name} | {target_disease}!\n")
            except subprocess.CalledProcessError as e:
                print(f"\nERROR: Training failed for {dataset_name} | {target_disease} with exit code {e.returncode}\n")
                # Continue to attempt training the others.

    print("Multi-Domain Batch Orchestration Finished!")

if __name__ == "__main__":
    main()
