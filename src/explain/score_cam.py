import torch
import torch.nn.functional as F

class ScoreCam:
    def __init__(self, model, target_layer, device='cpu'):
        """
        Custom, gradient-free Score-CAM implementation.
        Score-Weighted Visual Explanations for Convolutional Neural Networks.
        
        Args:
            model (nn.Module): Trained PyTorch model.
            target_layer (nn.Module): The convolutional layer to extract activations from.
            device (torch.device): Compute device for the model.
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        
        # Register forward hook to capture the activations from the target layer
        self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def attribute(self, input_tensor, target_class=0, chunk_size=32):
        """
        Generates the Score-CAM attribution heatmap.
        
        Args:
            input_tensor (torch.Tensor): Image tensor of shape [1, 3, H, W]
            target_class (int): The output class index to explain.
            chunk_size (int): Number of masked inputs to pass through the model simultaneously.
            
        Returns:
            torch.Tensor: The final upsampled Score-CAM attribution mask of shape [1, 1, H, W].
        """
        with torch.no_grad():
            # 1. Base forward pass to populate self.activations
            _ = self.model(input_tensor)
            
            if self.activations is None:
                raise ValueError("Failed to capture activations. Check if the target layer is executed in the forward pass.")
                
            activations = self.activations.detach()
            B, C, H, W = activations.size()
            
            assert B == 1, "Score-CAM implementation currently supports batch size of 1."
            
            # 2. Upsample activations to match original base image resolution
            input_h, input_w = input_tensor.size(2), input_tensor.size(3)
            upsampled_activations = F.interpolate(
                activations,
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False
            )
            
            # 3. Normalize each upsampled map into [0, 1] range
            acts_min = upsampled_activations.view(C, -1).min(dim=1, keepdim=True)[0].view(1, C, 1, 1)
            acts_max = upsampled_activations.view(C, -1).max(dim=1, keepdim=True)[0].view(1, C, 1, 1)
            normalized_activations = (upsampled_activations - acts_min) / (acts_max - acts_min + 1e-7)
            
            # 4. Generate masked images and calculate scores in chunks to prevent VRAM OOM
            scores = []
            for i in range(0, C, chunk_size):
                # Slice a chunk of masks: [chunk, 1, H, W]
                chunk_masks = normalized_activations[0, i:i+chunk_size].unsqueeze(1)
                
                # Multiply original image [1, 3, H, W] by masks [chunk, 1, H, W] -> [chunk, 3, H, W]
                masked_inputs = input_tensor.expand(chunk_masks.size(0), -1, -1, -1) * chunk_masks
                
                # Forward pass the masked images
                masked_outputs = self.model(masked_inputs)
                
                # Extract the logit targeted for our class
                if masked_outputs.dim() == 1 or masked_outputs.size(1) == 1:
                    chunk_scores = masked_outputs.view(-1)
                else:
                    chunk_scores = masked_outputs[:, target_class].view(-1)
                    
                scores.append(chunk_scores)
                
            scores = torch.cat(scores) # Shape: [C]
            
            # Softmax to scores to heavily weight the top features
            scores = F.softmax(scores, dim=0)
            
            # 5. Calculate final linear combination: sum(Score^c * Activation^c)
            # We apply it to the original sized activations first to save compute, then upsample the single result
            weighted_activations = (activations[0] * scores.view(C, 1, 1)).sum(dim=0, keepdim=True).unsqueeze(0)
            
            # 6. Apply ReLU to remove negative noise mathematically
            heatmap = F.relu(weighted_activations)
            
            # 7. Final upsample and normalization
            final_heatmap = F.interpolate(
                heatmap,
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False
            )
            
            heatmap_min = final_heatmap.min()
            heatmap_max = final_heatmap.max()
            final_heatmap = (final_heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-7)
            
            return final_heatmap.cpu()
            
    def remove_hook(self):
        """Clean up the forward hook."""
        self.hook_handle.remove()
