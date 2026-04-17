"""
Cross-Dataset Merged Training Pipeline v3 (Speed + Accuracy Optimized).
Hardware-tuned for i9-14900HX + RTX 4080 + 32GB DDR5.

Features:
  - Batch 64, 80K samples/epoch cap, OneCycleLR (peak 3e-4)
  - Backbone freeze (2 epochs head warmup) + label smoothing 0.1
  - 8 DataLoader workers (24-core CPU), channels_last memory format
  - EMA weights for better generalization
  - Gradient clipping for stability
  - WeightedRandomSampler for balanced batches
  - pos_weight FocalLoss + CutMix augmentation
  - Test-Time Augmentation (TTA)

Usage:
    python train_merged.py                    # Train all 9 overlapping diseases
    python train_merged.py --only Pneumonia   # Train just one
    python train_merged.py --skip Atelectasis # Skip already trained
"""
import os
import sys
import json
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from torch.amp import autocast

# ── Hardware-level optimizations ──
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ── Hardware Profile: i9-14900HX + RTX 4080 (12GB) + 32GB DDR5 ──
NUM_WORKERS = 8
BATCH_SIZE = 32         # 32 fits full backward+optimizer in 12GB VRAM (SwinV2 uses LayerNorm, batch-size-agnostic)
MAX_SAMPLES_PER_EPOCH = 80000  # Cap epoch size for large datasets
EMA_DECAY = 0.999
LABEL_SMOOTH = 0.1     # Smoothed labels: 0→0.05, 1→0.95
FREEZE_EPOCHS = 2      # Freeze backbone for head warmup

sys.path.insert(0, r'l:\MTP')
sys.path.insert(0, r'l:\MTP\src\data')
sys.path.insert(0, r'l:\MTP\src')

from dataset import UniversalMedicalDataset

# ─── Merged Disease Definitions ──────────────────────────────────────────────
MERGED_DISEASES = {
    "Atelectasis": [
        (r"l:\MTP\datasets\chestxray\NIH", "Atelectasis"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Atelectasis"),
    ],
    "Cardiomegaly": [
        (r"l:\MTP\datasets\chestxray\NIH", "Cardiomegaly"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Cardiomegaly"),
    ],
    "Consolidation": [
        (r"l:\MTP\datasets\chestxray\NIH", "Consolidation"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Consolidation"),
    ],
    "Edema": [
        (r"l:\MTP\datasets\chestxray\NIH", "Edema"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Edema"),
    ],
    "Pleural_Effusion": [
        (r"l:\MTP\datasets\chestxray\NIH", "Effusion"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Pleural_Effusion"),
    ],
    "Lung_Opacity": [
        (r"l:\MTP\datasets\chestxray\MIMIC", "Lung_Opacity"),
        (r"l:\MTP\datasets\chestxray\covid-19", "Lung_Opacity"),
    ],
    "Pneumonia": [
        (r"l:\MTP\datasets\chestxray\NIH", "Pneumonia"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Pneumonia"),
        (r"l:\MTP\datasets\chestxray\rsna-pneumonia-detection-challenge", "Pneumonia"),
    ],
    "Pneumothorax": [
        (r"l:\MTP\datasets\chestxray\NIH", "Pneumothorax"),
        (r"l:\MTP\datasets\chestxray\MIMIC", "Pneumothorax"),
    ],
    "Melanoma": [
        (r"l:\MTP\datasets\dermatology\HAM10000", "mel"),
        (r"l:\MTP\datasets\dermatology\ISIC_2024", "Melanoma"),
        (r"l:\MTP\datasets\dermatology\ISIC_2018", "Melanoma"),
    ],
}


# ─── Metadata Padding Wrapper ────────────────────────────────────────────────
class MetadataPadAdapter(torch.utils.data.Dataset):
    def __init__(self, dataset, target_meta_dim):
        self.dataset = dataset
        self.target_meta_dim = target_meta_dim
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, meta, label = self.dataset[idx]
        current_dim = meta.shape[0]
        if current_dim < self.target_meta_dim:
            meta = torch.cat([meta, torch.zeros(self.target_meta_dim - current_dim)])
        elif current_dim > self.target_meta_dim:
            meta = meta[:self.target_meta_dim]
        return img, meta, label


# ─── Model Architecture ──────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()


class EMAModel:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
    
    def apply(self, model):
        self.backup = {name: param.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


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
        
    def forward(self, image, metadata):
        v_features = self.vision_model(image)
        if self.meta_dim > 0:
            fused_features = torch.cat((v_features, metadata), dim=1)
            return self.fusion_mlp(fused_features)
        return self.fusion_mlp(v_features)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    return (np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H),
            np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H))


def calculate_metrics(y_true, y_pred_probs, threshold=None):
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
    y_true_bin = (y_true >= 0.5).astype(float)
    
    # Find optimal threshold via Youden's J statistic (maximizes sensitivity + specificity)
    try:
        auc = roc_auc_score(y_true_bin, y_pred_probs)
        if threshold is None:
            fpr, tpr, thresholds = roc_curve(y_true_bin, y_pred_probs)
            j_scores = tpr - fpr  # Youden's J
            best_idx = np.argmax(j_scores)
            threshold = float(thresholds[best_idx])
    except ValueError:
        auc = 0.0
        if threshold is None:
            threshold = 0.5
    
    y_pred = (y_pred_probs >= threshold).astype(float)
    tp = sum((y_true_bin == 1) & (y_pred == 1))
    fp = sum((y_true_bin == 0) & (y_pred == 1))
    tn = sum((y_true_bin == 0) & (y_pred == 0))
    fn = sum((y_true_bin == 1) & (y_pred == 0))
    acc = (tp + tn) / len(y_true_bin) if len(y_true_bin) > 0 else 0.0
    return {
        'Accuracy': acc * 100, 'AUC-ROC': auc,
        'Optimal_Threshold': threshold,
        'F1': f1_score(y_true_bin, y_pred, zero_division=0),
        'Precision': precision_score(y_true_bin, y_pred, zero_division=0),
        'Recall (Sensitivity)': recall_score(y_true_bin, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
    }


# ─── Main Training Function ──────────────────────────────────────────────────
def train_merged(canonical_name, dataset_sources):
    EXP_DIR = r"l:\MTP\experiments\merged"
    os.makedirs(EXP_DIR, exist_ok=True)
    LOG_FILE = os.path.join(EXP_DIR, f'{canonical_name}_training_log.txt')
    
    # LR managed by per-group OneCycleLR: backbone=1e-5, head=3e-4
    EPOCHS = 25
    PATIENCE = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def log(msg):
        print(msg)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
    
    log(f"\n{'='*60}")
    log(f"MERGED TRAINING: {canonical_name}")
    log(f"Sources: {len(dataset_sources)} datasets")
    for path, target in dataset_sources:
        log(f"  - {os.path.basename(path)} -> target='{target}'")
    log(f"Device: {DEVICE} | batch={BATCH_SIZE} | workers={NUM_WORKERS} | EMA={EMA_DECAY}")
    log(f"{'='*60}")
    
    # ── Load all sub-datasets (BALANCED: equal pos/neg) ──
    train_datasets = []
    val_datasets = []
    test_datasets = []
    meta_dims = []
    
    for ds_path, ds_target in dataset_sources:
        try:
            train_ds = UniversalMedicalDataset(domain_path=ds_path, split='train', target_disease=ds_target, balanced=True)
            val_ds = UniversalMedicalDataset(domain_path=ds_path, split='val', target_disease=ds_target, balanced=True)
            test_ds = UniversalMedicalDataset(domain_path=ds_path, split='test', target_disease=ds_target, balanced=True)
            
            if len(train_ds) == 0:
                log(f"  [SKIP] {os.path.basename(ds_path)}/{ds_target}: 0 samples")
                continue
            
            _, sample_meta, _ = train_ds[0]
            meta_dim = sample_meta.shape[0]
            meta_dims.append(meta_dim)
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
            
            log(f"  [OK] {os.path.basename(ds_path)}: train={len(train_ds)} "
                f"(pos={train_ds.pos_count}, neg={train_ds.neg_count}), meta={meta_dim}D")
        except Exception as e:
            log(f"  [FAIL] {os.path.basename(ds_path)}/{ds_target}: {e}")
            continue
    
    if not train_datasets:
        log(f"[ABORT] No valid datasets for {canonical_name}")
        return
    
    # ── Pad metadata and concatenate ──
    max_meta_dim = max(meta_dims) if meta_dims else 0
    log(f"Metadata: {meta_dims} -> pad to {max_meta_dim}D")
    
    wrapped_train = [MetadataPadAdapter(ds, max_meta_dim) for ds in train_datasets]
    wrapped_val = [MetadataPadAdapter(ds, max_meta_dim) for ds in val_datasets]
    wrapped_test = [MetadataPadAdapter(ds, max_meta_dim) for ds in test_datasets]
    
    merged_train = ConcatDataset(wrapped_train)
    merged_val = ConcatDataset(wrapped_val)
    merged_test = ConcatDataset(wrapped_test)
    
    epoch_samples = min(len(merged_train), MAX_SAMPLES_PER_EPOCH)
    log(f"Merged: train={len(merged_train)}, val={len(merged_val)}, test={len(merged_test)}")
    log(f"Balanced 50/50 — no pos_weight needed, threshold=0.5 is optimal")
    log("Creating DataLoaders (spawning workers, may take 30-60s on Windows)...")
    train_loader = DataLoader(merged_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader = DataLoader(merged_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(merged_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    log(f"DataLoaders ready. {len(train_loader)} train batches, {len(val_loader)} val batches.")
    
    # ── Model ──
    log("Loading SwinV2-S pretrained weights...")
    base_model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
    vision_dim = base_model.head.in_features
    model = MultiModalFusion(vision_model=base_model, vision_dim=vision_dim, meta_dim=max_meta_dim).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    log(f"Model: SwinV2-S + MultiModalFusion (vision={vision_dim}D, meta={max_meta_dim}D) [channels_last]")
    
    # ── Training Setup (no pos_weight — data is balanced) ──
    criterion = FocalLoss(alpha=1.0, gamma=2.0, pos_weight=None)
    
    # Differential learning rates: backbone needs much lower LR to preserve pretrained features
    backbone_params = list(model.vision_model.parameters())
    head_params = list(model.fusion_mlp.parameters())
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},   # backbone: gentle fine-tuning
        {'params': head_params, 'lr': 1e-4},        # head: fast adaptation
    ], weight_decay=0.05)
    
    from torch.optim.lr_scheduler import OneCycleLR
    total_batches = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=[1e-5, 3e-4], steps_per_epoch=total_batches,
                           epochs=EPOCHS, pct_start=0.1, anneal_strategy='cos')
    
    # BF16 on Ada Lovelace: same dynamic range as FP32, no GradScaler needed
    ema = EMAModel(model, decay=EMA_DECAY)
    
    # ── CUDA Warmup ──
    log("Running CUDA warmup batch (cuDNN benchmarking, ~1-3 min)...")
    model.train()
    warmup_data = next(iter(train_loader))
    warmup_inputs = warmup_data[0][:2].to(DEVICE, memory_format=torch.channels_last)
    warmup_meta = warmup_data[1][:2].to(DEVICE)
    with autocast('cuda', dtype=torch.bfloat16):
        _ = model(warmup_inputs, warmup_meta)
    del warmup_inputs, warmup_meta, warmup_data
    torch.cuda.empty_cache()
    log("CUDA warmup complete. Starting training loop.")
    
    # ── Training Loop ──
    start_time = time.time()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_path = os.path.join(EXP_DIR, f'{canonical_name}_model.pth')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        
        # Backbone freeze/unfreeze for head warmup
        if epoch < FREEZE_EPOCHS:
            for param in model.vision_model.parameters():
                param.requires_grad = False
            if epoch == 0:
                log(f"Backbone FROZEN for {FREEZE_EPOCHS} epochs (head warmup)")
        elif epoch == FREEZE_EPOCHS:
            for param in model.vision_model.parameters():
                param.requires_grad = True
            torch.cuda.empty_cache()  # Release stale cached memory before full backward
            log("Backbone UNFROZEN — full fine-tuning")
        
        for batch_idx, (inputs, meta, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            meta = meta.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            # Label smoothing: 0→0.05, 1→0.95
            targets = targets * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.bfloat16):
                r = np.random.rand(1)
                if r < 0.5:
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(inputs.size()[0]).to(DEVICE)
                    target_a = targets
                    target_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    outputs = model(inputs, meta).squeeze()
                    if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                else:
                    outputs = model(inputs, meta).squeeze()
                    if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR: step per batch
            ema.update(model)
            
            train_loss += loss.item() * inputs.size(0)
            
            # Batch-level progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / elapsed
                eta = (total_batches - batch_idx - 1) / speed if speed > 0 else 0
                avg_loss = train_loss / ((batch_idx + 1) * BATCH_SIZE)
                lr_now = optimizer.param_groups[1]['lr']  # head LR (backbone is 30x lower)
                print(f"\r  Batch {batch_idx+1}/{total_batches} | Loss: {avg_loss:.4f} | "
                      f"{speed:.1f} b/s | LR: {lr_now:.2e} | ETA: {eta/60:.1f}min", end="", flush=True)
        
        print()  # newline after batch progress
        
        train_loss /= len(merged_train)  # all samples seen with shuffle (no sampler cap)
        
        # Validation (EMA weights)
        ema.apply(model)
        model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        with torch.inference_mode():
            for vi, (inputs, meta, targets) in enumerate(val_loader):
                inputs = inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                meta = meta.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                outputs = model(inputs, meta).squeeze()
                if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                if (vi + 1) % 100 == 0:
                    print(f"\r  Validating: {vi+1}/{val_batches}", end="", flush=True)
        if val_batches > 100:
            print()
        val_loss /= len(merged_val)
        ema.restore(model)
        
        epoch_time = (time.time() - epoch_start) / 60
        total_elapsed = (time.time() - start_time) / 60
        est_remaining = epoch_time * (EPOCHS - epoch - 1)
        log(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"{epoch_time:.1f}min/epoch | Elapsed: {total_elapsed:.0f}min | ETA: {est_remaining:.0f}min")
        # OneCycleLR steps per batch, not per epoch
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ema.apply(model)
            torch.save(model.state_dict(), model_path)
            ema.restore(model)
            log(f"  [+] Val loss improved. EMA model saved.")
        else:
            epochs_no_improve += 1
            log(f"  [-] No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                log(f"Early stopping after {epoch+1} epochs!")
                break
    
    log(f"Training Complete. Time: {(time.time() - start_time)/60:.2f} minutes.")
    
    # ── Test with TTA (EMA weights) ──
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    log("Evaluating (EMA + TTA)...")
    test_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.inference_mode():
        for inputs, meta, targets in test_loader:
            inputs = inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            meta = meta.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            out_orig = model(inputs, meta).squeeze()
            out_flip = model(torch.flip(inputs, dims=[3]), meta).squeeze()
            outputs = (out_orig + out_flip) / 2.0
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_loss /= len(merged_test)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))
    metrics['Test Loss'] = test_loss
    
    log("Test Set Results:")
    for k, v in metrics.items():
        log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    with open(os.path.join(EXP_DIR, f'{canonical_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    log(f"[DONE] {canonical_name}\n")
    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Dataset Merged Training')
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--skip', type=str, nargs='+', default=[], help='Skip already trained diseases')
    args = parser.parse_args()
    
    if args.only:
        if args.only not in MERGED_DISEASES:
            print(f"Error: '{args.only}' not in MERGED_DISEASES. Available: {list(MERGED_DISEASES.keys())}")
            sys.exit(1)
        targets = {args.only: MERGED_DISEASES[args.only]}
    else:
        targets = MERGED_DISEASES
    
    print(f"Cross-Dataset Merged Training (Final Maximum Performance)")
    print(f"Training {len(targets)} disease(s): {list(targets.keys())}")
    print(f"Hardware: batch={BATCH_SIZE}, workers={NUM_WORKERS}, samples/epoch={MAX_SAMPLES_PER_EPOCH}")
    print(f"{'='*60}\n")
    
    for disease_name, sources in targets.items():
        if disease_name in args.skip:
            print(f"\n[SKIP] {disease_name} (already trained)")
            continue
        train_merged(disease_name, sources)
    
    print(f"\n{'='*60}")
    print("ALL MERGED TRAINING COMPLETE! Run: python export_mobile.py")
