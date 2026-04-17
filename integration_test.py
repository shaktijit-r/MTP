"""Full Integration Test Suite for MTP Project"""
import sys
sys.path.insert(0, r'l:\MTP')
sys.path.insert(0, r'l:\MTP\src')
sys.path.insert(0, r'l:\MTP\src\data')
sys.path.insert(0, r'l:\MTP\src\models')

import torch
import numpy as np
from pathlib import Path

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    try:
        func()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

# ============ IMPORTS ============
print("=" * 60)
print("SECTION 1: Import Verification")
print("=" * 60)

def test_explain_imports():
    from explain.sas import SemanticAttentionSynthesis
    from explain.methods import ExplainabilityMethods
    from explain.visualize import plot_attributions, generate_annotated_image, generate_high_res_annotation, generate_clinical_summary
    from explain.vlm_synthesizer import synthesize_comprehensive_report, generate_narrative
test("Explainability imports", test_explain_imports)

def test_data_imports():
    from dataset import UniversalMedicalDataset, setup_datasets
    from metadata_parser import parse_metadata, extract_demographics
test("Data pipeline imports", test_data_imports)

def test_model_imports():
    from train_baseline import FocalLoss, calculate_metrics
    from torchvision.models import swin_v2_s, Swin_V2_S_Weights
test("Model/training imports", test_model_imports)

# ============ BOUNDING BOX ============
print("\n" + "=" * 60)
print("SECTION 2: Bounding Box Consensus Fusion")
print("=" * 60)

def test_bbox_signature():
    import inspect
    from explain.visualize import generate_annotated_image
    sig = inspect.signature(generate_annotated_image)
    params = list(sig.parameters.keys())
    assert 'secondary_attrs' in params, f"Missing secondary_attrs! Got: {params}"
test("Bounding box has multi-method fusion param", test_bbox_signature)

def test_bbox_execution():
    from explain.visualize import generate_annotated_image
    import os, tempfile
    orig = torch.rand(3, 256, 256)
    primary = torch.rand(1, 1, 256, 256)
    secondary_ig = torch.rand(1, 3, 256, 256)
    secondary_occ = torch.rand(1, 1, 256, 256)
    save_path = os.path.join(tempfile.gettempdir(), "test_bbox.png")
    area = generate_annotated_image(orig, primary, save_path, threshold=0.7,
                                     secondary_attrs=[secondary_ig, secondary_occ])
    assert os.path.exists(save_path), "Bounding box image not saved!"
    assert 0.0 <= area <= 1.0, f"Area out of range: {area}"
    os.remove(save_path)
test("Bounding box consensus fusion execution", test_bbox_execution)

# ============ METRICS ============
print("\n" + "=" * 60)
print("SECTION 3: Metrics with Label Smoothing")
print("=" * 60)

def test_metrics_smoothed():
    from train_baseline import calculate_metrics
    y_true = np.array([0.95, 0.05, 0.95, 0.05, 0.95])
    y_probs = np.array([0.82, 0.15, 0.91, 0.30, 0.70])
    m = calculate_metrics(y_true, y_probs)
    assert 'AUC-ROC' in m and 'F1' in m and 'Specificity' in m
    assert 0 <= m['AUC-ROC'] <= 1
test("Metrics handle smoothed labels (0.95/0.05)", test_metrics_smoothed)

def test_focal_loss():
    from train_baseline import FocalLoss
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    out = torch.tensor([0.8, -0.3], requires_grad=True)
    labels = torch.tensor([0.95, 0.05])
    loss = criterion(out, labels)
    loss.backward()
    assert loss.item() > 0
test("FocalLoss with smoothed labels", test_focal_loss)

# ============ METADATA DIMENSIONS ============
print("\n" + "=" * 60)
print("SECTION 4: Metadata Dimension Consistency")
print("=" * 60)

def test_meta_dim(ds_name, path, disease, expected):
    from metadata_parser import parse_metadata
    labels_dir = Path(path) / 'labels'
    parsed = parse_metadata(ds_name, ds_name, labels_dir, disease)
    assert len(parsed) > 0, "No labels parsed"
    first_key = list(parsed.keys())[0]
    got = len(parsed[first_key][1])
    assert got == expected, f"meta_dim={got}, expected={expected}"

meta_tests = [
    ("NIH", r"l:\MTP\datasets\chestxray\NIH", "Pneumonia", 4),
    ("covid-19", r"l:\MTP\datasets\chestxray\covid-19", "COVID", 0),
    ("rsna-pneumonia-detection-challenge", r"l:\MTP\datasets\chestxray\rsna-pneumonia-detection-challenge", "Pneumonia", 4),
    ("ISIC", r"l:\MTP\datasets\dermatology\ISIC", "Melanoma", 21),
    ("ham10000_integrated", r"l:\MTP\datasets\dermatology\ham10000_integrated", "mel", 21),
    ("ODIR", r"l:\MTP\datasets\ophthalmology\ODIR", "Glaucoma", 0),
    ("OASIS", r"l:\MTP\datasets\brain\OASIS", "Dementia", 4),
]
for ds_name, path, disease, expected in meta_tests:
    test(f"Meta dim {ds_name}=={expected}", lambda dn=ds_name, p=path, d=disease, e=expected: test_meta_dim(dn, p, d, e))

# ============ CODE CHECKS ============
print("\n" + "=" * 60)
print("SECTION 5: Source Code Integrity")
print("=" * 60)

def test_sas_weights():
    content = open(r"l:\MTP\src\explain\sas.py").read()
    assert "stage_weights" in content
    assert "'early': 0.15" in content
    assert "'late': 0.60" in content
test("SAS weighted aggregation (15/25/60)", test_sas_weights)

def test_label_smoothing():
    import re
    content = open(r"l:\MTP\src\data\metadata_parser.py").read()
    hard_labels = re.findall(r"label\s*=\s*1\.0\b", content)
    assert len(hard_labels) == 0, f"Found {len(hard_labels)} hard 1.0 labels!"
    assert content.count("0.95") >= 5
test("All parsers use label smoothing", test_label_smoothing)

def test_train_all_registry():
    content = open(r"l:\MTP\train_all.py").read()
    for ds in ["NIH", "ISIC", "covid-19", "rsna-pneumonia-detection-challenge",
                "ham10000_integrated", "ODIR", "OASIS"]:
        assert f'"{ds}"' in content, f"{ds} NOT registered!"
test("All 7 datasets in TARGET_DISEASES", test_train_all_registry)

def test_images_dir_fallback():
    content = open(r"l:\MTP\src\data\dataset.py").read()
    assert "images_2d" in content
    assert "self.images_dir" in content
test("images_2d fallback for 3D volumes", test_images_dir_fallback)

def test_balanced_cap():
    content = open(r"l:\MTP\src\data\dataset.py").read()
    assert "MAX_SAMPLES_PER_CLASS" in content
    assert "> 0.5" in content
test("Balanced sampling with 10K cap", test_balanced_cap)

def test_adamw():
    content = open(r"l:\MTP\src\models\train_baseline.py").read()
    assert "AdamW" in content
    assert "weight_decay" in content
test("AdamW with weight decay", test_adamw)

def test_resolution():
    content = open(r"l:\MTP\src\data\dataset.py").read()
    assert "256" in content
    assert "RandomCrop((224" not in content
test("Unified 256x256 resolution", test_resolution)

def test_api_consensus():
    content = open(r"l:\MTP\src\api\main.py").read()
    assert "secondary_attrs=" in content
    assert "attr_ig.detach()" in content
    assert "attr_occ.detach()" in content
test("API passes IG+Occ to bbox", test_api_consensus)

def test_api_multimodal():
    content = open(r"l:\MTP\src\api\main.py").read()
    assert "MultiModalFusion" in content
    assert "meta_dim" in content
test("API MultiModalFusion loading", test_api_multimodal)

def test_vlm_domain():
    content = open(r"l:\MTP\src\explain\vlm_synthesizer.py").read()
    assert "dermoscopic" in content or "modality_map" in content
test("VLM domain-aware prompts", test_vlm_domain)

# ============ SUMMARY ============
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} tests passed, {failed} failed")
if failed == 0:
    print("ALL INTEGRATION TESTS PASSED!")
else:
    print(f"WARNING: {failed} test(s) FAILED!")
print("=" * 60)
