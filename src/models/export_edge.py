import os
import argparse
import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

def export_to_onnx(model_path, target_disease, output_dir):
    """
    Converts a standard PyTorch Swin-T classification model into an ONNX graph 
    optimzed for Edge Compute (NPUs, iOS CoreML, Android TFLite).
    """
    DEVICE = torch.device('cpu') 
    
    print(f"Loading Swin-T V2 from {model_path}...")
    
    # 1. Initialize the exact architecture
    model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
    num_ftrs = model.head.in_features
    # Our architecture uses a single output logit for BCEWithLogitsLoss
    model.head = nn.Linear(num_ftrs, 1)
    
    # 2. Load the trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Cannot export.")
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Create dummy input matching the Swin-T input space (BatchSize=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    
    # 4. Define dynamic axes (Allows the exported model to accept dynamic batch sizes)
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    output_path = os.path.join(output_dir, f"{target_disease}_edge.onnx")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Compiling PyTorch execution graph to ONNX...")
    
    # 5. Export
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=14,          # Opset 14 is highly stable for Transformers
        do_constant_folding=True,  # Optimize convolutions and batch norms
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"SUCCESS: Edge Model exported to {output_path}")
    print("This .onnx file can now be directly embedded into an iOS/Android application.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export PyTorch Swin-T to ONNX for Edge App')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth weights')
    parser.add_argument('--target_disease', type=str, required=True, help='Name of the disease (e.g., Pneumonia)')
    parser.add_argument('--output_dir', type=str, default='experiments/edge_models', help='Output directory')
    args = parser.parse_args()
    
    export_to_onnx(args.model_path, args.target_disease, args.output_dir)
