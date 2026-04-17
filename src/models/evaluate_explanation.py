import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader

import sys
# Allow importing from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import ChestXray14Dataset
from explain.methods import ExplainabilityMethods
from explain.visualize import plot_attributions

def load_trained_model(model_path, device):
    """Loads the pre-trained ResNet-18 baseline model."""
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Standard single logit output that we trained
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Explainability Evaluation on: {DEVICE}")
    
    ROOT_DIR = 'dataset'
    EXP_DIR = 'experiments/baseline'
    MODEL_PATH = os.path.join(EXP_DIR, 'model.pth')
    OUT_DIR = 'experiments/explainability/visualizations'
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading test dataset...")
    test_dataset = ChestXray14Dataset(root_dir=ROOT_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Loading trained baseline model...")
    model = load_trained_model(MODEL_PATH, DEVICE)
    
    # Initialize our Captum wrapper
    explainer = ExplainabilityMethods(model, device=DEVICE)
    
    num_samples_to_explain = 5
    explained_count = 0
    
    print(f"Beginning Explainability Analysis for {num_samples_to_explain} samples...")
    for idx, (image, label) in enumerate(test_loader):
        image = image.to(DEVICE)
        
        # To make Grad-CAM work correctly with BCEWithLogits, we need gradients to flow
        image.requires_grad = True 
        
        with torch.no_grad():
            output = model(image).squeeze()
            if output.dim() == 0: output = output.unsqueeze(0)
            prob = torch.sigmoid(output).item()
        
        # Determine actual classification (threshold 0.5)
        pred_class = 1 if prob >= 0.5 else 0
        true_class = int(label.item())
        
        # For this demonstration, we specifically look for images the model predicted as Pneumonia (or very confident negatives) 
        # to ensure the heatmaps have interesting gradients to trace.
        
        # We'll just take the first few images to prove the pipeline works end-to-end
        print(f"  -> Generating explanations for sample {idx} (True: {true_class}, Pred: {pred_class})")
        
        attributions = {}
        
        # 1. Grad-CAM
        try:
            attributions['Grad-CAM'] = explainer.generate_gradcam(image, target_class=0)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            
        # 2. Integrated Gradients
        try:
            attributions['Integrated Gradients'] = explainer.generate_integrated_gradients(image, target_class=0, n_steps=25)
        except Exception as e:
            print(f"IG failed: {e}")
            
        # 3. Occlusion
        try:
            # Masking 15x15 pixel windows with a stride of 8
            attributions['Occlusion'] = explainer.generate_occlusion(image, target_class=0, 
                                                                     sliding_window_shapes=(3, 15, 15), 
                                                                     strides=(3, 8, 8))
        except Exception as e:
            print(f"Occlusion failed: {e}")
            
        # Save visualization overlay
        save_path = os.path.join(OUT_DIR, f"sample_{idx}_true{true_class}_pred{pred_class}.png")
        
        image_cpu_nodeltas = image.detach().cpu().squeeze(0)
        
        plot_attributions(
            original_image=image_cpu_nodeltas,
            attributions_dict=attributions,
            save_path=save_path,
            true_label=true_class,
            pred_prob=prob
        )
        
        explained_count += 1
        if explained_count >= num_samples_to_explain:
            break

    print(f"Explainability visualizations saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
