import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_t
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import setup_datasets

class RAGImageRetriever:
    """
    Retrieval-Augmented Diagnosing class that stores pre-calculated Swin-T latent embeddings
    and retrieves K-nearest visually/mathematically similar cases for a given input.
    """
    def __init__(self, model, index_path="experiments/rag/index.pt", meta_path="experiments/rag/meta.json", device='cuda'):
        self.device = device
        self.model = model.to(self.device).eval()
        self.index_path = index_path
        self.meta_path = meta_path
        
        # Remove the classification head to extract the latent embedding (dim=768)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        self.embeddings = None
        self.metadata = None
        
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load_index()
            
    def _extract_embedding(self, input_tensor):
        """Extracts the 1D latent vector from the Swin-T backbone."""
        with torch.no_grad():
            x = self.feature_extractor(input_tensor)
            # Output of Swin-T features before head is typically shape [B, C] -> [1, 768]
            x = torch.flatten(x, 1)
            # L2 Normalize for Cosine Similarity
            x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def build_index(self, dataloader):
        """
        Runs the entire dataset through the model and saves its embeddings and metadata.
        Must be called offline.
        """
        print("Building RAG Image Index...")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        all_embeddings = []
        all_meta = []
        
        for images, labels, paths in tqdm(dataloader, desc="Extracting Latents"):
            images = images.to(self.device)
            emb = self._extract_embedding(images)
            all_embeddings.append(emb.cpu())
            
            for i in range(len(paths)):
                # We save path and diagnosis for context mapping
                all_meta.append({
                    "path": paths[i],
                    "label": int(labels[i].item()),
                    "diagnosis": "Pneumonia" if labels[i].item() == 1 else "Normal"
                })
                
        self.embeddings = torch.cat(all_embeddings, dim=0) # [N, 768]
        self.metadata = all_meta
        
        torch.save(self.embeddings, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f)
            
        print(f"Successfully built RAG index with {len(all_meta)} patients.")

    def load_index(self):
        """Loads the precomputed PyTorch tensor bank and JSON metadata."""
        self.embeddings = torch.load(self.index_path, map_location='cpu')
        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)

    def retrieve_similar(self, input_tensor, k=3):
        """
        Finds the top-K most similar historical patients using Cosine Similarity.
        """
        if self.embeddings is None:
            raise ValueError("RAG index not built or loaded.")
            
        # Extract query vector
        q = self._extract_embedding(input_tensor).cpu() # [1, 768]
        
        # Calculate Cosine Similarity (Dot product of L2 normalized vectors)
        similarities = torch.mm(q, self.embeddings.t()).squeeze(0) # [N]
        
        # Get top-K indices
        top_scores, top_indices = torch.topk(similarities, k)
        
        results = []
        for i in range(k):
            idx = int(top_indices[i].item())
            score = float(top_scores[i].item())
            meta = self.metadata[idx]
            results.append({
                "similarity_score": score,
                "label": meta["label"],
                "diagnosis": meta["diagnosis"],
                "path": meta["path"]
            })
            
        return results

if __name__ == "__main__":
    # Script to build the index offline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading datasets...")
    # Load training dataset ONLY so the model acts as a "memory bank" of what it was taught on
    train_loader, _, _ = setup_datasets(batch_size=64, get_paths=True)
    
    print("Loading Swin-T model...")
    model_path = os.path.abspath('experiments/baseline/model.pth')
    model = swin_v2_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 1)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("WARNING: Using untrained Swin-T since model.pth was not found.")
        
    rag = RAGImageRetriever(model, device=device)
    rag.build_index(train_loader)
