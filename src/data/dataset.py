import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.data.metadata_parser import parse_metadata

class UniversalMedicalDataset(Dataset):
    def __init__(self, domain_path, split='train', target_disease='Pneumonia', get_paths=False, seed=42, balanced=False):
        """
        Universal Dataset for all medical imaging modalities.
        When balanced=True, undersamples the majority class to match minority class count.
        """
        self.domain_path = Path(domain_path)
        self.split = split
        self.target_disease = target_disease
        self.get_paths = get_paths
        
        self.dataset_name = self.domain_path.name
        self.domain_name = self.domain_path.parent.name
        
        # Aggressive augmentation for training (proven to boost medical imaging accuracy)
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Parse Labels
        labels_dir = self.domain_path / "labels"
        parsed_labels = parse_metadata(self.domain_name, self.dataset_name, labels_dir, target_disease)
        
        # Cross-reference with actual images on disk
        images_dir = self.domain_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {self.domain_path}")
        
        self.images_dir = images_dir
        disk_files = set(os.listdir(images_dir))
        
        valid_files = []
        for filename, label in parsed_labels.items():
            if filename in disk_files:
                valid_files.append((filename, label))
                
        # Fuzzy matching fallback
        if len(valid_files) == 0 and len(parsed_labels) > 0 and len(disk_files) > 0:
            print("Warning: Exact filename match failed. Attempting fuzzy matching without extensions...")
            disk_base_to_full = {Path(f).stem: f for f in disk_files}
            for filename, val in parsed_labels.items():
                base = Path(filename).stem
                if base in disk_base_to_full:
                    valid_files.append((disk_base_to_full[base], val))
                    
        if len(valid_files) == 0:
            print(f"Warning: No valid overlapping images and labels found in {self.domain_path}")
            self.samples = []
            self.pos_count = 0
            self.neg_count = 0
            return
            
        # Deterministic Shuffle for Split
        random.seed(seed)
        valid_files.sort()
        random.shuffle(valid_files)
        
        total = len(valid_files)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)
        
        if split == 'train':
            self.samples = valid_files[:train_end]
        elif split == 'val':
            self.samples = valid_files[train_end:val_end]
        else:
            self.samples = valid_files[val_end:]
        
        # Balance classes by undersampling majority class
        if balanced and len(self.samples) > 0:
            pos_samples = [s for s in self.samples if s[1][0] > 0.5]
            neg_samples = [s for s in self.samples if s[1][0] <= 0.5]
            min_count = min(len(pos_samples), len(neg_samples))
            if min_count > 0:
                random.seed(seed)  # Deterministic undersampling
                pos_samples = random.sample(pos_samples, min_count)
                neg_samples = random.sample(neg_samples, min_count)
                self.samples = pos_samples + neg_samples
                random.shuffle(self.samples)
        
        # Compute class distribution
        self.pos_count = sum(1 for s in self.samples if s[1][0] > 0.5)
        self.neg_count = len(self.samples) - self.pos_count
        
        print(f"[{self.dataset_name}] {split.upper()} Split: {len(self.samples)} samples "
              f"({self.pos_count} pos, {self.neg_count} neg, ratio={self.pos_count/max(self.neg_count,1):.3f})"
              + (f" [BALANCED]" if balanced else ""))

    def get_sample_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler.
        Minority class samples get higher weight so batches are balanced.
        """
        total = self.pos_count + self.neg_count
        if total == 0 or self.pos_count == 0 or self.neg_count == 0:
            return [1.0] * len(self.samples)
        
        w_pos = total / (2.0 * self.pos_count)   # Higher weight for minority
        w_neg = total / (2.0 * self.neg_count)    # Lower weight for majority
        
        weights = []
        for _, (label, _) in self.samples:
            weights.append(w_pos if label > 0.5 else w_neg)
        return weights

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, (label, meta) = self.samples[idx]
        img_path = self.images_dir / filename
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        meta_tensor = torch.tensor(meta, dtype=torch.float32)
        
        if self.get_paths:
            return image, meta_tensor, label_tensor, str(img_path)
        return image, meta_tensor, label_tensor

def setup_datasets(batch_size=32, domain_path='l:/MTP/datasets/chestxray/NIH', target_disease='Pneumonia', get_paths=False):
    train_dataset = UniversalMedicalDataset(domain_path=domain_path, split='train', target_disease=target_disease, get_paths=get_paths)
    val_dataset = UniversalMedicalDataset(domain_path=domain_path, split='val', target_disease=target_disease, get_paths=get_paths)
    test_dataset = UniversalMedicalDataset(domain_path=domain_path, split='test', target_disease=target_disease, get_paths=get_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
