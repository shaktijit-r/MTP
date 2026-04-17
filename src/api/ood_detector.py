import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import swin_v2_s
import numpy as np
import os

class ClinicalOODDetector:
    """
    Mahalanobis Distance Out-Of-Distribution (OOD) Detector.
    Extracts the latent 768-D vision vectors from a trusted Medical Image dataset,
    computes the multivariate Mean and Covariance Matrix, and flags new incoming
    images that mathematically deviate too far from standard radiological geometry.
    """
    def __init__(self, device='cuda'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the base Swin Backbone (Without the classification head)
        self.encoder = swin_v2_s()
        self.encoder.head = nn.Identity() # Expose the raw 768-D vector
        self.encoder.to(self.device)
        self.encoder.eval()
        
        self.mean_vector = None
        self.inv_covariance = None
        self.threshold = 0.0 # Chi-Square 95% threshold to be set during fitting
        
    def _extract_features(self, dataloader):
        """Extracts 768-D vectors for the entire passed dataloader."""
        features = []
        with torch.no_grad():
            from tqdm import tqdm
            for inputs, _, _ in tqdm(dataloader, desc="Extracting OOD Manifold Vectors"):
                inputs = inputs.to(self.device)
                vec = self.encoder(inputs).cpu().numpy()
                features.append(vec)
        return np.vstack(features)
        
    def fit(self, dataloader, save_dir='models/ood'):
        """Computes the Mean and Covariance matrix from the training set."""
        print("Extracting feature manifold from standard dataset...")
        features = self._extract_features(dataloader) # [N, 768]
        
        print("Computing Normal Distribution Parameters...")
        self.mean_vector = np.mean(features, axis=0)
        
        # Add epsilon to diagonal to ensure numerical stability (invertibility)
        epsilon = 1e-6
        cov_matrix = np.cov(features, rowvar=False) + np.eye(features.shape[1]) * epsilon
        self.inv_covariance = np.linalg.inv(cov_matrix)
        
        # Calculate Mahalanobis distance for all training samples to define a 99th percentile threshold
        print("Calculating baseline variances...")
        distances = []
        for feat in features:
            diff = feat - self.mean_vector
            dist = np.sqrt(np.dot(np.dot(diff, self.inv_covariance), diff.T))
            distances.append(dist)
            
        # Set threshold to legally reject the top 1% weirdest/corrupted images
        self.threshold = np.percentile(distances, 99.0)
        print(f"OOD Safety Threshold Established at MD: {self.threshold:.2f}")
        
        # Save structural parameters
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/mean_vector.npy', self.mean_vector)
        np.save(f'{save_dir}/inv_covariance.npy', self.inv_covariance)
        np.save(f'{save_dir}/threshold.npy', np.array([self.threshold]))
        print(f"OOD Mathematical Matrices successfully saved to {save_dir}/")
        
    def load(self, load_dir='models/ood'):
        """Loads pre-computed OOD matrices for API Inference."""
        try:
            self.mean_vector = np.load(f'{load_dir}/mean_vector.npy')
            self.inv_covariance = np.load(f'{load_dir}/inv_covariance.npy')
            self.threshold = float(np.load(f'{load_dir}/threshold.npy')[0])
            print("Loaded Clinical OOD Protection Matrix.")
        except Exception as e:
            print(f"Warning: OOD models missing from {load_dir}. Please run 'fit()' first.")
            
    def is_ood(self, image_tensor):
        """
        Receives a 1xCxHxW PyTorch tensor.
        Returns (True/False, Mahalanobis Score) where True means REJECT image.
        """
        if self.mean_vector is None:
            raise ValueError("OOD Detector has not loaded its mathematical baseline.")
            
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            vec = self.encoder(image_tensor).cpu().numpy().squeeze()
            
        diff = vec - self.mean_vector
        mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, self.inv_covariance), diff.T))
        
        # Return True if image is too statistically weird (Non-Xray artifact)
        reject = mahalanobis_dist > self.threshold
        return reject, mahalanobis_dist
