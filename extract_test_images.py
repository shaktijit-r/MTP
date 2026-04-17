"""
Extract High-Confidence Test Images for Presentation.
Creates l:\MTP\test_images\ with one positive and one negative image per disease.
Files are named: {Disease}_positive.png, {Disease}_negative.png
"""
import os, sys, shutil, torch, torch.nn as nn
from torchvision.models import swin_v2_s
from torchvision import transforms
from PIL import Image
from pathlib import Path

sys.path.insert(0, r'l:\MTP')
sys.path.insert(0, r'l:\MTP\src\data')
sys.path.insert(0, r'l:\MTP\src')

from verify_launch import TARGET_DISEASES, DATASET_PATHS
from dataset import UniversalMedicalDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path(r"l:\MTP\test_images")
OUT_DIR.mkdir(exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiModalFusion(nn.Module):
    def __init__(self, vision_model, vision_dim, meta_dim, num_classes=1):
        super().__init__()
        self.vision_model = vision_model
        self.meta_dim = meta_dim
        self.vision_model.head = nn.Identity()
        if self.meta_dim > 0:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(vision_dim + meta_dim, 256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, num_classes))
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(vision_dim, num_classes))
    def forward(self, image, metadata=None):
        v = self.vision_model(image)
        if self.meta_dim > 0:
            if metadata is None:
                metadata = torch.zeros(image.shape[0], self.meta_dim, device=image.device)
            return self.fusion_mlp(torch.cat((v, metadata), dim=1))
        return self.fusion_mlp(v)

def load_model(disease):
    import glob
    paths = glob.glob(os.path.join(r"l:\MTP\experiments", "**", f"{disease}_model.pth"), recursive=True)
    if not paths:
        return None
    sd = torch.load(paths[0], map_location=DEVICE, weights_only=False)
    base = swin_v2_s(weights=None)
    vdim = base.head.in_features
    mdim = 0
    for k in sd:
        if 'fusion_mlp.0.weight' in k:
            mdim = sd[k].shape[1] - vdim; break
    if mdim < 0: mdim = 0
    model = MultiModalFusion(base, vdim, mdim)
    try:
        model.load_state_dict(sd)
    except:
        return None
    model.to(DEVICE).eval()
    return model

# Deduplicate diseases across datasets — pick best dataset per disease
UNIQUE_DISEASES = {}
# Priority: merged > single-dataset (some diseases trained on merged data)
for ds_name, diseases in TARGET_DISEASES.items():
    for d in diseases:
        if d in ('No_Finding', 'Normal', 'Support_Devices', 'Pleural_Other', 'Other', 'mel'):
            continue  # Skip non-disease targets
        if d == 'Effusion':
            d = 'Pleural_Effusion'  # Mapped name
        if d not in UNIQUE_DISEASES:
            UNIQUE_DISEASES[d] = (ds_name, DATASET_PATHS[ds_name])

print(f"Extracting test images for {len(UNIQUE_DISEASES)} diseases...\n")

for disease, (ds_name, ds_path) in UNIQUE_DISEASES.items():
    pos_file = OUT_DIR / f"{disease}_positive.png"
    neg_file = OUT_DIR / f"{disease}_negative.png"
    
    if pos_file.exists() and neg_file.exists():
        print(f"  [Skip] {disease} — already extracted")
        continue
    
    print(f"  [{ds_name}] {disease}...", end=" ", flush=True)
    
    model = load_model(disease)
    if model is None:
        print("NO MODEL")
        continue
    
    try:
        dataset = UniversalMedicalDataset(domain_path=ds_path, split='test', target_disease=disease)
    except Exception as e:
        print(f"DATASET ERROR: {e}")
        continue
    
    best_pos = (None, -1, 0.0)   # (idx, path, confidence)
    best_neg = (None, -1, 1.0)
    
    max_scan = min(len(dataset), 500)  # Cap at 500 samples for speed
    
    for i in range(max_scan):
        try:
            _, meta_t, label = dataset[i]
            val = float(label)
            fname, _ = dataset.samples[i]
            raw_path = dataset.images_dir / fname
            
            img = Image.open(raw_path).convert("RGB")
            inp = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            meta = meta_t.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if model.meta_dim > 0:
                    out = model(inp, meta)
                else:
                    out = model(inp)
                out = out.squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                prob = torch.sigmoid(out).item()
            
            # True Positive: label=1, model says positive with high confidence
            if val > 0.5 and prob > best_pos[2]:
                best_pos = (i, raw_path, prob)
            # True Negative: label=0, model says negative with high confidence  
            elif val <= 0.5 and prob < best_neg[2]:
                best_neg = (i, raw_path, prob)
            
            # Early exit if we found very high confidence for both
            if best_pos[2] > 0.90 and best_neg[2] < 0.10:
                break
                
        except Exception:
            continue
    
    found = []
    if best_pos[1] is not None and best_pos[1] != -1:
        img = Image.open(best_pos[1]).convert("RGB").resize((512, 512))
        img.save(pos_file)
        found.append(f"POS={best_pos[2]:.1%}")
    if best_neg[1] is not None and best_neg[1] != -1:
        img = Image.open(best_neg[1]).convert("RGB").resize((512, 512))
        img.save(neg_file)
        found.append(f"NEG={best_neg[2]:.1%}")
    
    print(", ".join(found) if found else "NO SAMPLES FOUND")

print(f"\nDone! Images saved to: {OUT_DIR}")
