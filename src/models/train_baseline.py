import os
import json
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from dataset import UniversalMedicalDataset
import numpy as np
from torch.amp import autocast

# ── Hardware-level optimizations ──
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ── Hardware Profile: i9-14900HX + RTX 4080 (12GB) + 32GB DDR5 ──
NUM_WORKERS = 8         # 24-core CPU: use 8 DataLoader workers
BATCH_SIZE = 32         # 32 fits full backward+optimizer in 12GB VRAM (SwinV2 uses LayerNorm, batch-size-agnostic)
MAX_SAMPLES_PER_EPOCH = 80000  # Cap epoch size for large datasets
EMA_DECAY = 0.999       # Exponential Moving Average decay rate
LABEL_SMOOTH = 0.1      # Smoothed labels: 0→0.05, 1→0.95
FREEZE_EPOCHS = 2       # Freeze backbone for head warmup


class FocalLoss(nn.Module):
    """Focal Loss with class-aware pos_weight for imbalanced datasets."""
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
    """Exponential Moving Average of model weights for better generalization."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
    
    def apply(self, model):
        """Apply EMA weights to model (for evaluation)."""
        self.backup = {name: param.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


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
    
    f1 = f1_score(y_true_bin, y_pred, zero_division=0)
    precision = precision_score(y_true_bin, y_pred, zero_division=0)
    recall_sensitivity = recall_score(y_true_bin, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'Accuracy': acc * 100,
        'AUC-ROC': auc,
        'Optimal_Threshold': threshold,
        'F1': f1,
        'Precision': precision,
        'Recall (Sensitivity)': recall_sensitivity,
        'Specificity': specificity,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description='Train SwinV2-S on specific disease')
    parser.add_argument('--domain_path', type=str, default='l:/MTP/datasets/chestxray/NIH')
    parser.add_argument('--target_disease', type=str, default='Pneumonia')
    args = parser.parse_args()
    
    target_disease = args.target_disease
    
    # ── Hyperparameters (Maximum Accuracy) ──
    # LR managed by per-group OneCycleLR: backbone=1e-5, head=3e-4
    EPOCHS = 25
    PATIENCE = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE} for disease: {target_disease}")

    # Paths
    DOMAIN_PATH = args.domain_path
    dataset_name = os.path.basename(os.path.normpath(DOMAIN_PATH))
    EXP_DIR = f'experiments/{dataset_name}'
    LOG_FILE = os.path.join(EXP_DIR, f'{target_disease}_training_log.txt')
    os.makedirs(EXP_DIR, exist_ok=True)
    
    def log(msg):
        print(msg)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')

    log(f"Hardware profile: batch={BATCH_SIZE}, workers={NUM_WORKERS}, "
        f"samples/epoch={MAX_SAMPLES_PER_EPOCH}, EMA={EMA_DECAY}")

    # Load datasets (BALANCED: equal pos/neg)
    try:
        train_dataset = UniversalMedicalDataset(domain_path=DOMAIN_PATH, split='train', target_disease=target_disease, balanced=True)
        val_dataset = UniversalMedicalDataset(domain_path=DOMAIN_PATH, split='val', target_disease=target_disease, balanced=True)
        test_dataset = UniversalMedicalDataset(domain_path=DOMAIN_PATH, split='test', target_disease=target_disease, balanced=True)
        
        if len(train_dataset) == 0:
            log(f"Dataset {dataset_name} has 0 valid samples for {target_disease}. Skipping.")
            return

        log("Creating DataLoaders (spawning workers, may take 30-60s on Windows)...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        log(f"DataLoaders ready. {len(train_loader)} train batches, {len(val_loader)} val batches.")
        log(f"Balanced 50/50 — no pos_weight needed, threshold=0.5 is optimal")
    except FileNotFoundError as e:
        log(f"Dataset error: {e}")
        return

    # ── Model Setup ──
    log("Loading SwinV2-S pretrained weights...")
    base_model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)

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

    sample_meta_tensor = train_dataset[0][1]
    calculated_meta_dim = sample_meta_tensor.shape[0]
    log(f"EHR Vector Width: {calculated_meta_dim}-D")

    model = MultiModalFusion(vision_model=base_model, vision_dim=base_model.head.in_features, meta_dim=calculated_meta_dim).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    log(f"Model ready on {DEVICE} [channels_last]")

    # ── Loss + Optimizer (no pos_weight — data is balanced) ──
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
    log("Running CUDA warmup (cuDNN benchmarking, ~1-3 min)...")
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
        train_loss /= len(train_dataset)  # all balanced samples seen with shuffle
        
        # Validation (using EMA weights)
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
                
        val_loss /= len(val_loader.dataset)
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
            # Save EMA weights (they generalize better)
            ema.apply(model)
            model_path = os.path.join(EXP_DIR, f'{target_disease}_model.pth')
            torch.save(model.state_dict(), model_path)
            ema.restore(model)
            log(f"  [+] Validation loss improved. EMA model saved to {model_path}")
        else:
            epochs_no_improve += 1
            log(f"  [-] No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                log(f"Early stopping triggered after {epoch+1} epochs!")
                break

    log(f"Training Complete. Time: {(time.time() - start_time)/60:.2f} minutes.")
    
    # ── Test Evaluation with TTA (using saved EMA weights) ──
    model_path = os.path.join(EXP_DIR, f'{target_disease}_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    log("Evaluating on Test Set (EMA weights + TTA: original + hflip)...")
    test_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.inference_mode():
        for inputs, meta, targets in test_loader:
            inputs = inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            meta = meta.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            # TTA: Average original + horizontal flip
            out_orig = model(inputs, meta).squeeze()
            out_flip = model(torch.flip(inputs, dims=[3]), meta).squeeze()
            outputs = (out_orig + out_flip) / 2.0
            
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    test_loss /= len(test_loader.dataset)
    
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))
    metrics['Test Loss'] = test_loss
    
    log("Test Set Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            log(f"  {k}: {v:.4f}")
        else:
            log(f"  {k}: {v}")
            
    with open(os.path.join(EXP_DIR, f'{target_disease}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.array([[metrics['TN'], metrics['FP']], 
                       [metrics['FN'], metrics['TP']]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'No {target_disease}', target_disease],
                    yticklabels=[f'No {target_disease}', target_disease])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{target_disease} SwinV2-S Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(EXP_DIR, f'{target_disease}_confusion_matrix.png'))
        plt.close()
        log("Confusion matrix image saved.")
    except ImportError:
        log("Matplotlib/seaborn not installed, skipping confusion matrix.")

if __name__ == '__main__':
    main()
