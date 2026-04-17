import torch
from captum.attr import LayerGradCam, IntegratedGradients, Occlusion, GuidedGradCam
from captum.attr import LayerAttribution
from explain.score_cam import ScoreCam
from explain.sas import SemanticAttentionSynthesis

class ExplainabilityMethods:
    def __init__(self, model, device='cpu'):
        """
        Wrapper class for initializing and running different Captum attribution methods.
        
        Args:
            model (nn.Module): The trained PyTorch model.
            device (torch.device): The device the model is on.
        """
        self.model = model
        self.device = device
        self.model.eval()

    def generate_gradcam(self, input_tensor, target_class=0, target_layer=None):
        """
        Generates Grad-CAM attribution using Captum.
        
        Args:
            input_tensor (torch.Tensor): Image tensor of shape [1, 3, H, W]
            target_class (int): The output class index to explain (0 for our single-logit BCE).
            target_layer (nn.Module): The convolutional layer to attach Grad-CAM to.
                                      Defaults to the last conv layer of a ResNet.
        Returns:
            torch.Tensor: The upsampled Grad-CAM attribution mask.
        """
        if target_layer is None:
            # For torchvision.models.resnet18, 'layer4' is the final bottleneck block
            target_layer = self.model.layer4[-1].conv2
            
        layer_gc = LayerGradCam(self.model, target_layer)
        
        # Attribute relative to the single output logit (index 0)
        attributions = layer_gc.attribute(input_tensor, target=target_class)
        
        # Upsample attribution to match original image size (224x224)
        upsampled_attr = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])
        return upsampled_attr.detach().cpu()

    def generate_sas(self, input_tensor):
        """
        Generates Semantic Attention Synthesis (SAS) attribution.
        Extracts the native feature magnitude matrix natively from the Swin Transformer.
        
        Args:
            input_tensor (torch.Tensor): Image tensor of shape [1, 3, H, W]
        Returns:
            torch.Tensor: The SAS attribution mask.
        """
        sas = SemanticAttentionSynthesis(self.model, device=self.device)
        # Call the standalone generator function
        attributions = sas.generate_attention_map(input_tensor)
        
        return attributions
        
    def generate_integrated_gradients(self, input_tensor, target_class=0, n_steps=50):
        """
        Generates Integrated Gradients attribution using Captum.
        
        Args:
            input_tensor (torch.Tensor): Image tensor of shape [1, 3, H, W]
            target_class (int): The output class index to explain.
            n_steps (int): The number of steps for the integral approximation.
        Returns:
            torch.Tensor: The Integrated Gradients attribution mask.
        """
        ig = IntegratedGradients(self.model)
        
        # Baseline is a black image (all zeros) by default
        baseline = torch.zeros_like(input_tensor).to(self.device)
        
        attributions, delta = ig.attribute(input_tensor, 
                                           baselines=baseline, 
                                           target=target_class, 
                                           n_steps=n_steps, 
                                           return_convergence_delta=True)
        return attributions.detach().cpu()

    def generate_occlusion(self, input_tensor, target_class=0, sliding_window_shapes=(3, 15, 15), strides=(3, 8, 8)):
        """
        Generates Occlusion attribution using Captum.
        
        Args:
            input_tensor (torch.Tensor): Image tensor of shape [1, 3, H, W]
            target_class (int): The output class index to explain.
            sliding_window_shapes (tuple): Shape of the occluding patch (channels, height, width).
            strides (tuple): Strides of the sliding window.
        Returns:
            torch.Tensor: The Occlusion attribution mask.
        """
        # For ResNet18 BCEWithLogits validation, we must ensure sliding window covers 3 channels
        occlusion = Occlusion(self.model)
        
        # Baseline is a black image (all zeros) for the occluded region
        baseline = torch.zeros_like(input_tensor).to(self.device)
        
        attributions = occlusion.attribute(input_tensor,
                                           strides=strides,
                                           target=target_class,
                                           sliding_window_shapes=sliding_window_shapes,
                                           baselines=baseline)
        return attributions.detach().cpu()
