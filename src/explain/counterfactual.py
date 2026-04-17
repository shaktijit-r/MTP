import cv2
import numpy as np

def generate_counterfactual(original_img_path, sas_tensor, output_path, threshold=0.6):
    """
    Generative Counterfactual 'What-If' Inpainting.
    Takes the original image and the precise SAS attention matrix.
    Uses the hottest regions of the attention map as a 'damage mask' 
    and mathematically reconstructs continuous healthy tissue in the void 
    using Navier-Stokes based fluid dynamics inpainting.
    
    Args:
        original_img_path (str): Path to the 224x224 RGB image.
        sas_tensor (torch.Tensor): The [1, 1, H, W] normalized SAS map (0.0 to 1.0).
        output_path (str): Where to save the 'cured' X-Ray.
        threshold (float): Above this threshold, a pixel is considered 'diseased'.
    """
    # 1. Load original image
    img = cv2.imread(original_img_path)
    if img is None:
        raise ValueError(f"Could not load image for counterfactual generation: {original_img_path}")
        
    # Resize to ensure match if needed
    img = cv2.resize(img, (224, 224))
    
    # 2. Process SAS mask
    sas_np = sas_tensor.squeeze().cpu().numpy()
    
    # Create the binary diseased mask
    # 255 = diseased/inpaint target, 0 = healthy tissue to keep
    mask = (sas_np >= threshold).astype(np.uint8) * 255
    
    # Erode the mask slightly so we only inpaint the absolute core dense regions
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # 3. Generative Navier-Stokes Inpainting
    # Inpaint radius determines how far healthy pixels are sampled.
    radius = 7
    cured_img = cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    
    # 4. Optional: Add a subtle overlay so clinicians know they are looking at a synthetic
    # We add a tiny green watermark in the top right
    cv2.putText(cured_img, "SYNTHETIC: COUNTERFACTUAL 'CURED'", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
    # 5. Save the result
    cv2.imwrite(output_path, cured_img)
