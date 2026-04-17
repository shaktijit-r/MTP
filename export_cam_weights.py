"""
Export classification weights for CAM (Class Activation Mapping) heatmaps.
Extracts fusion_mlp first-layer weights (vision portion only) per disease.
Saves as compact binary files for mobile app consumption.
"""
import os, sys, glob, struct
import torch
import torch.nn as nn
from torchvision.models import swin_v2_s, swin_v2_t
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

VALID_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Emphysema", "Enlarged_Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung_Lesion", "Lung_Opacity",
    "Mass", "Nodule", "Pleural_Effusion", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
    "COVID", "Viral_Pneumonia",
    "AMD", "Cataract", "Diabetes", "Diabetic_Retinopathy", "Glaucoma", "Hypertension", "Myopia",
    "Melanoma", "akiec", "bcc", "bkl", "df", "nv", "vasc",
    "Dementia",
]


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobile_app", "assets", "cam_weights")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Exporting CAM weights for {len(VALID_DISEASES)} diseases")
    print(f"Output: {out_dir}\n")

    vision_dim = 768  # SwinV2-S output dimension
    success = 0

    for disease in VALID_DISEASES:
        # Find model weights
        merged_path = os.path.join(base_dir, "merged", f"{disease}_model.pth")
        if os.path.exists(merged_path):
            model_path = merged_path
        else:
            paths = glob.glob(os.path.join(base_dir, "**", f"{disease}_model.pth"), recursive=True)
            if not paths:
                print(f"  [SKIP] {disease} — no weights found")
                continue
            model_path = paths[0]

        sd = torch.load(model_path, map_location='cpu', weights_only=True)

        # Extract fusion_mlp first Linear layer weights
        # When meta_dim>0: fusion_mlp.0.weight (Linear is first)
        # When meta_dim=0: fusion_mlp.1.weight (Dropout is first, Linear is second)
        w_key = None
        for key in sd:
            if 'fusion_mlp.0.weight' in key:
                w_key = key
                break

        if w_key is None:
            for key in sd:
                if 'fusion_mlp.1.weight' in key:
                    w_key = key
                    break

        if w_key is None:
            # Fallback: simple model with head directly (SwinV2-T legacy)
            for key in sd:
                if key == 'head.weight':
                    w_key = key
                    break

        if w_key is None:
            print(f"  [SKIP] {disease} — no classification weights found")
            continue

        weights = sd[w_key]  # [out_features, in_features]
        # Take only the vision dimension columns (first 768)
        cam_weights = weights[:, :vision_dim]  # [256, 768] or [1, 768]

        # For CAM, we need a single weight per channel.
        # Sum across output neurons to get channel importance: [768]
        channel_weights = cam_weights.sum(dim=0).float().numpy()  # [768]

        # Save as raw float32 binary
        out_path = os.path.join(out_dir, f"{disease}_cam.bin")
        channel_weights.tofile(out_path)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  [OK] {disease}_cam.bin ({size_kb:.1f} KB, {len(channel_weights)} channels)")
        success += 1

    total_kb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f.endswith('.bin')) / 1024
    print(f"\nDone: {success}/{len(VALID_DISEASES)} weights exported ({total_kb:.0f} KB total)")


if __name__ == "__main__":
    main()
