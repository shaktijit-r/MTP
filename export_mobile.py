"""
Export trained PyTorch models to ONNX format (.onnx) for ONNX Runtime Mobile.
Prioritizes merged cross-dataset models over single-dataset models.

Usage:
    python export_mobile.py                      # Export all 35 valid diseases
    python export_mobile.py --only Pneumonia     # Export just one
"""
import torch
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'data'))

import torch.nn as nn
from torchvision.models import swin_v2_s, swin_v2_t

DEVICE = torch.device('cpu')  # Export always on CPU

# ─── Canonical list of 35 valid diseases for the mobile app ──────────────────
VALID_DISEASES = [
    # Chest X-Ray (18)
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Emphysema", "Enlarged_Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung_Lesion", "Lung_Opacity",
    "Mass", "Nodule", "Pleural_Effusion", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
    # COVID X-Ray (2)
    "COVID", "Viral_Pneumonia",
    # Ophthalmology (7)
    "AMD", "Cataract", "Diabetes", "Diabetic_Retinopathy", "Glaucoma", "Hypertension", "Myopia",
    # Dermatology (7)
    "Melanoma", "akiec", "bcc", "bkl", "df", "nv", "vasc",
    # Neurology (1)
    "Dementia",
]


# ─── Model Architecture (must match training) ────────────────────────────────
class MultiModalFusion(nn.Module):
    def __init__(self, vision_model, vision_dim, meta_dim, num_classes=1):
        super(MultiModalFusion, self).__init__()
        self.vision_model = vision_model
        self.meta_dim = meta_dim
        self.vision_model.head = nn.Identity()
        if self.meta_dim > 0:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(vision_dim + meta_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(vision_dim, num_classes)
            )
    def forward(self, image, metadata=None):
        v_features = self.vision_model(image)
        if self.meta_dim > 0 and metadata is not None:
            fused_features = torch.cat((v_features, metadata), dim=1)
            return self.fusion_mlp(fused_features)
        return self.fusion_mlp(v_features)


def load_model_for_disease(disease_name):
    """
    Loads the best .pth weights for the requested disease target.
    Priority order:
      1. experiments/merged/{disease}_model.pth  (cross-dataset merged model)
      2. experiments/**/{disease}_model.pth       (single-dataset model)
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    
    # Priority 1: Merged model
    merged_path = os.path.join(base_dir, "merged", f"{disease_name}_model.pth")
    if os.path.exists(merged_path):
        model_path = merged_path
        print(f"  [Merged model found] {model_path}")
    else:
        # Priority 2: Single-dataset model (glob search)
        search_path = os.path.join(base_dir, "**", f"{disease_name}_model.pth")
        model_paths = glob.glob(search_path, recursive=True)
        if not model_paths:
            return None
        model_path = model_paths[0]
        print(f"  [Single-dataset model] {model_path}")
    
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    
    base_model = swin_v2_s(weights=None)
    vision_dim = base_model.head.in_features
    
    # Detect metadata dimension from state dict
    meta_dim = 0
    for key in state_dict:
        if 'fusion_mlp.0.weight' in key:
            total_input = state_dict[key].shape[1]
            meta_dim = total_input - vision_dim
            break
    if meta_dim < 0:
        meta_dim = 0
    
    model = MultiModalFusion(vision_model=base_model, vision_dim=vision_dim, meta_dim=meta_dim)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # Legacy fallback (SwinV2-Tiny models from earlier training)
        model = swin_v2_t(weights=None)
        model.head = nn.Linear(model.head.in_features, 1)
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


def export_to_mobile(disease_target, output_dir="mobile_app/assets"):
    """Export a single disease model to .onnx mobile format."""
    print(f"\nLoading best weights for {disease_target}...")
    model = load_model_for_disease(disease_target)
    
    if model is None:
        print(f"  [SKIP] No trained weights found for {disease_target}")
        return False
        
    model.eval()
    model.to('cpu')
    
    meta_dim = getattr(model, 'meta_dim', 0)
    print(f"  meta_dim={meta_dim}")

    print("  Converting to Float16 (Half Precision)...")
    model = model.half()

    # Wrap model to accept ONLY image input (mobile has no EHR metadata)
    # Also outputs spatial feature map for CAM explainability
    class MobileWrapper(torch.nn.Module):
        def __init__(self, inner_model, md):
            super().__init__()
            self.inner_model = inner_model
            self.md = md
        def forward(self, image):
            image = image.half()  # FP32 from bridge -> FP16 for model weights
            
            # Split SwinV2 forward to extract spatial features before pooling
            vm = self.inner_model.vision_model
            spatial = vm.features(image)       # [B, H, W, C]
            spatial = vm.norm(spatial)          # LayerNorm
            spatial = vm.permute(spatial)       # [B, C, H, W] = [B, 768, 8, 8]
            feature_map = spatial.float()       # Save for CAM output
            
            # Continue with pooling + classification
            pooled = vm.avgpool(spatial)        # [B, C, 1, 1]
            flat = vm.flatten(pooled)           # [B, C]
            
            # Fusion MLP
            if self.md > 0:
                meta = torch.zeros(image.shape[0], self.md, dtype=flat.dtype, device=flat.device)
                out = self.inner_model.fusion_mlp(torch.cat((flat, meta), dim=1))
            else:
                out = self.inner_model.fusion_mlp(flat)
            
            return out.float(), feature_map    # (logits, spatial_features)

    wrapper = MobileWrapper(model, meta_dim)
    wrapper.eval()

    print("  Preparing dummy inputs...")
    dummy_img = torch.randn(1, 3, 256, 256)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{disease_target}_mobile.onnx")
        
        print("  Exporting to ONNX format (with feature map output)...")
        torch.onnx.export(
            wrapper,
            dummy_img,
            out_path,
            input_names=['image'],
            output_names=['logits', 'feature_map'],
            dynamic_axes={
                'image': {0: 'batch'},
                'logits': {0: 'batch'},
                'feature_map': {0: 'batch'},
            },
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,  # Legacy exporter handles FP16 SwinV2 correctly
        )
        
        # Consolidate external data files into single self-contained .onnx file
        import onnx
        data_path = out_path + ".data"
        if os.path.exists(data_path):
            print("  Consolidating external data into single file...")
            onnx_model = onnx.load(out_path, load_external_data=True)
            onnx.save_model(onnx_model, out_path, save_as_external_data=False)
            # Remove leftover .data file
            if os.path.exists(data_path):
                os.remove(data_path)
        
        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"  [OK] {disease_target}_mobile.onnx ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Export error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Export models to mobile format')
    parser.add_argument('--only', type=str, default=None,
                        help='Export only this specific disease')
    args = parser.parse_args()
    
    if args.only:
        targets = [args.only]
    else:
        targets = VALID_DISEASES
    
    print(f"Exporting {len(targets)} disease model(s) to ONNX format (.onnx)")
    print(f"Output directory: mobile_app/assets/")
    print(f"{'='*60}\n")
    
    success = 0
    failed = []
    
    for disease in targets:
        if export_to_mobile(disease):
            success += 1
        else:
            failed.append(disease)
    
    print(f"\n{'='*60}")
    print(f"Export Complete: {success}/{len(targets)} succeeded")
    if failed:
        print(f"Failed: {failed}")
    
    # Calculate total size
    total_size = 0
    for disease in VALID_DISEASES:
        onnx_path = os.path.join("mobile_app", "assets", f"{disease}_mobile.onnx")
        if os.path.exists(onnx_path):
            total_size += os.path.getsize(onnx_path)
    print(f"Total model size: {total_size / (1024*1024*1024):.2f} GB")
