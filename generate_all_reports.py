"""
Batch Orchestration for Fully Automated Explainability Report Generation.
Generates 1 Positive and 1 Negative report per disease across all 52 targets.
"""
import os
import sys
import uuid
import torch
import torch.nn as nn
from torchvision.models import swin_v2_s, swin_v2_t
from torchvision import transforms
from PIL import Image
import numpy as np
import traceback
from tqdm import tqdm

sys.path.insert(0, r'l:\MTP')
sys.path.insert(0, r'l:\MTP\src')
sys.path.insert(0, r'l:\MTP\src\data')

from verify_launch import TARGET_DISEASES, DATASET_PATHS
from dataset import UniversalMedicalDataset
from explain.methods import ExplainabilityMethods
from explain.visualize import plot_attributions, generate_annotated_image, generate_high_res_annotation
from explain.vlm_synthesizer import init_vlm, synthesize_comprehensive_report
from api.pdf_generator import generate_clinical_pdf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    def forward(self, image, metadata=None):
        v_features = self.vision_model(image)
        if self.meta_dim > 0 and metadata is not None:
            fused_features = torch.cat((v_features, metadata), dim=1)
            return self.fusion_mlp(fused_features)
        return self.fusion_mlp(v_features)

class PureVisionWrapper(nn.Module):
    """
    Wraps the MultiModalFusion model so Captum pure-vision 
    explainability methods can run natively while still utilizing the fixed 
    EHR metadata tensor for the specific patient.
    """
    def __init__(self, model, fixed_metadata):
        super().__init__()
        self.model = model
        self.fixed_metadata = fixed_metadata
    
    @property
    def features(self):
        # Expose the internal vision model features to SAS mapper!
        if hasattr(self.model, 'vision_model'):
            return self.model.vision_model.features
        return self.model.features
        
    def forward(self, img):
        # Check if model is MultiModalFusion with metadata support
        if hasattr(self.model, 'meta_dim') and self.model.meta_dim > 0:
            batch_size = img.shape[0]
            meta_expanded = self.fixed_metadata.expand(batch_size, -1)
            return self.model(img, meta_expanded)
        elif hasattr(self.model, 'meta_dim'):
            # MultiModalFusion with meta_dim=0
            return self.model(img)
        else:
            # Legacy plain SwinTransformer
            return self.model(img)

def load_model_for_disease(disease_name):
    """Loads the highest performing .pth weights for the requested target."""
    import glob
    search_path = os.path.join(r"l:\MTP\experiments", "**", f"{disease_name}_model.pth")
    model_paths = glob.glob(search_path, recursive=True)
    
    if not model_paths:
        return None
        
    model_path = model_paths[0]
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    base_model = swin_v2_s(weights=None)
    vision_dim = base_model.head.in_features
    
    meta_dim = 0
    for key in state_dict:
        if 'fusion_mlp.0.weight' in key:
            total_input = state_dict[key].shape[1]
            meta_dim = total_input - vision_dim
            break
    if meta_dim < 0: meta_dim = 0
    
    model = MultiModalFusion(vision_model=base_model, vision_dim=vision_dim, meta_dim=meta_dim)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # Legacy fallback
        model = swin_v2_t(weights=None)
        model.head = nn.Linear(model.head.in_features, 1)
        model.load_state_dict(state_dict)
        
    model = model.to(DEVICE)
    model.eval()
    return model

def find_samples(dataset, model):
    """Iterates through the dataset and finds the first TRUE POSITIVE and first TRUE NEGATIVE sample."""
    pos_idx, neg_idx = None, None
    print("      [Searching for True Positive / True Negative samples...]")
    for i in range(len(dataset)):
        try:
            _, meta_tensor_raw, label = dataset[i]
            val = float(label)
            
            if (val > 0.5 and pos_idx is not None) or (val <= 0.5 and neg_idx is not None):
                continue
                
            filename, _ = dataset.samples[i]
            raw_img_path = dataset.images_dir / filename
            img_pil = Image.open(raw_img_path).convert("RGB")
            
            input_t = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
            meta_t = meta_tensor_raw.unsqueeze(0).to(DEVICE)
            
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'meta_dim') and model.meta_dim > 0:
                    out = model(input_t, meta_t)
                else:
                    out = model(input_t)
                out = out.squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                prob = torch.sigmoid(out).item()
                
            if val > 0.5 and prob >= 0.5 and pos_idx is None:
                pos_idx = i
            elif val <= 0.5 and prob < 0.5 and neg_idx is None:
                neg_idx = i
                
            if pos_idx is not None and neg_idx is not None:
                break
        except Exception as e:
            continue
            
    # Fallback if no perfect true pos/neg found
    if pos_idx is None or neg_idx is None:
        for i in range(len(dataset)):
            try:
                _, _, label = dataset[i]
                val = float(label)
                if val > 0.5 and pos_idx is None: pos_idx = i
                elif val <= 0.5 and neg_idx is None: neg_idx = i
                if pos_idx is not None and neg_idx is not None: break
            except: continue
            
    return pos_idx, neg_idx

def mc_dropout_inference(model, img_tensor, meta_tensor, num_passes=30):
    """Returns the ensemble predictions and standard deviation."""
    def enable_dropout(m):
        if type(m) == nn.Dropout or "Drop" in str(type(m)):
            m.train()
    
    model.apply(enable_dropout)
    probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            if isinstance(model, PureVisionWrapper):
                out = model(img_tensor)
            elif hasattr(model, 'meta_dim') and model.meta_dim > 0:
                out = model(img_tensor, meta_tensor)
            else:
                # Legacy plain SwinTransformer or meta_dim=0
                out = model(img_tensor)
            out = out.squeeze()
            if out.dim() == 0: out = out.unsqueeze(0)
            probs.append(torch.sigmoid(out).item())
            
    model.eval()
    return float(np.mean(probs)), float(np.std(probs))

def main():
    print("====== INITIATING FULL MULTI-DOMAIN REPORT GENERATION ======")
    out_dir = r"l:\MTP\experiments\reports"
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("[1] Initializing Template Engine...")
    init_vlm(DEVICE)

    for domain_name, target_list in TARGET_DISEASES.items():
        domain_path = DATASET_PATHS[domain_name]
        
        for tgt in target_list:
            print(f"\n>> Processing: {domain_name} | {tgt}")
            
            # Check if output PDFs already exist
            pos_pdf = os.path.join(out_dir, f"{tgt}_Positive.pdf")
            neg_pdf = os.path.join(out_dir, f"{tgt}_Negative.pdf")
            
            if os.path.exists(pos_pdf) and os.path.exists(neg_pdf):
                print(f"  [Skipping] Reports already exist for {tgt}.")
                continue
                
            # 1. Load Model
            model = load_model_for_disease(tgt)
            if model is None:
                print(f"  [Error] No trained weights found for {tgt}!")
                continue
                
            # 2. Extract Data
            try:
                dataset = UniversalMedicalDataset(domain_path=domain_path, split='test', target_disease=tgt)
                pos_idx, neg_idx = find_samples(dataset, model)
            except Exception as e:
                print(f"  [Error] Failed to load dataset splits for {tgt}: {e}")
                continue
                
            cases_to_run = []
            if pos_idx is not None: cases_to_run.append(('Positive', pos_idx))
            if neg_idx is not None: cases_to_run.append(('Negative', neg_idx))
            
            if not cases_to_run:
                print(f"  [Error] Could not locate any valid Positive/Negative ground truth samples in {tgt} test split.")
                continue
                
            for label_name, idx in cases_to_run:
                print(f"  --> Simulating Explanations for [Target: {tgt}] [Class: {label_name}]")
                try:
                    # Get the raw metadata tensor from the dataset
                    _, meta_tensor_raw, _ = dataset[idx]
                    
                    # Load the ORIGINAL untouched image directly from disk
                    # (dataset.__getitem__ returns ImageNet-normalized tensors which
                    #  clip/wrap when converted back to PIL, causing color corruption)
                    filename, _ = dataset.samples[idx]
                    raw_img_path = dataset.images_dir / filename
                    img_pil = Image.open(raw_img_path).convert("RGB")
                    img_resized = img_pil.resize((256, 256))
                    
                    # Build the properly normalized model input from the clean PIL image
                    input_t = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
                    input_t.requires_grad = True
                    
                    # Metadata formulation
                    meta_t = meta_tensor_raw.unsqueeze(0).to(DEVICE)
                    
                    # Wrap model for Captum Explainability integration
                    wrapped_model = PureVisionWrapper(model, meta_t)
                    
                    # Probabilities via ensemble dropout
                    prob, unct = mc_dropout_inference(model, input_t, meta_t)
                    
                    # Generate temporary visual artifacts
                    req_id = f"{tgt}_{label_name}_{uuid.uuid4().hex[:4]}"
                    base_path = os.path.join(temp_dir, req_id)
                    
                    orig_path = f"{base_path}_orig.png"
                    gc_path = f"{base_path}_gc.png"
                    ig_path = f"{base_path}_ig.png"
                    occ_path = f"{base_path}_occ.png"
                    bb_path = f"{base_path}_bb.png"
                    hr_path = f"{base_path}_hr.png"
                    cam_path = f"{base_path}_cam.png"
                    
                    img_resized.save(orig_path)
                    
                    # Run Deep Explainers
                    explainer = ExplainabilityMethods(wrapped_model, device=DEVICE)
                    
                    # SAS routing mapping
                    gradcam_attr = explainer.generate_sas(input_t)
                    unbatched_vis = transforms.ToTensor()(img_resized).to(DEVICE)
                    plot_attributions(unbatched_vis, {"SAS": gradcam_attr}, gc_path, "Unknown", prob)
                    
                    # High Resolution IG Tracing
                    ig_attr = explainer.generate_integrated_gradients(input_t, target_class=0, n_steps=18)
                    plot_attributions(unbatched_vis, {"IG": ig_attr}, ig_path, "Unknown", prob)
                    generate_high_res_annotation(unbatched_vis, ig_attr.detach(), hr_path, threshold=0.88)
                    
                    # Occlusion Perturbation
                    occ_attr = explainer.generate_occlusion(input_t, target_class=0, sliding_window_shapes=(3, 15, 15), strides=(3, 8, 8))
                    plot_attributions(unbatched_vis, {"OCC": occ_attr}, occ_path, "Unknown", prob)
                    
                    # Generate Bounding box fusion
                    bb_area, spatial_regions = generate_annotated_image(unbatched_vis, gradcam_attr.detach(), bb_path, threshold=0.7, secondary_attrs=[ig_attr.detach(), occ_attr.detach()])

                    # CAM (Class Activation Mapping) — extract spatial features before pooling
                    try:
                        vm = model.vision_model if hasattr(model, 'vision_model') else model
                        with torch.no_grad():
                            spatial = vm.features(input_t)
                            spatial = vm.norm(spatial)
                            spatial = vm.permute(spatial)  # [1, 768, 8, 8]
                        # Get classification weights
                        w_key = None
                        for key in model.state_dict():
                            if 'fusion_mlp.0.weight' in key:
                                w_key = key; break
                        if w_key is None:
                            for key in model.state_dict():
                                if 'fusion_mlp.1.weight' in key:
                                    w_key = key; break
                        if w_key:
                            weights = model.state_dict()[w_key][:, :768]  # [out, 768]
                            channel_w = weights.sum(dim=0)  # [768]
                            cam_map = torch.zeros(8, 8, device=DEVICE)
                            for c in range(768):
                                cam_map += channel_w[c] * spatial[0, c]
                            cam_map = torch.relu(cam_map)
                            if cam_map.max() > 0:
                                cam_map = cam_map / cam_map.max()
                            # Upsample to 256x256 and create overlay
                            import cv2 as cv2_cam
                            cam_np = cam_map.cpu().numpy()
                            cam_resized = cv2_cam.resize(cam_np, (256, 256), interpolation=cv2_cam.INTER_LINEAR)
                            cam_colored = cv2_cam.applyColorMap((cam_resized * 255).astype(np.uint8), cv2_cam.COLORMAP_JET)
                            orig_bgr = cv2_cam.cvtColor((np.transpose(unbatched_vis.cpu().numpy(), (1,2,0)) * 255).clip(0,255).astype(np.uint8), cv2_cam.COLOR_RGB2BGR)
                            cam_overlay = cv2_cam.addWeighted(orig_bgr, 0.6, cam_colored, 0.4, 0)
                            cv2_cam.imwrite(cam_path, cam_overlay)
                            print(f"      [V] CAM heatmap generated")
                        else:
                            cam_path = None
                    except Exception as cam_err:
                        print(f"      [!] CAM generation skipped: {cam_err}")
                        cam_path = None
                    
                    # Template-based Synthesis (grounded in model outputs)
                    narratives = synthesize_comprehensive_report(
                        prob, spatial_regions, tgt,
                        area_pct=bb_area, uncertainty=unct
                    )
                    
                    # Assemble Output Document
                    target_pdf = os.path.join(out_dir, f"{tgt}_{label_name}.pdf")
                    generate_clinical_pdf(
                        patient_id=f"TEST-{req_id.upper()}",
                        prediction_prob=prob,
                        uncertainty_score=unct,
                        original_img_path=orig_path,
                        gradcam_img_path=gc_path,
                        ig_img_path=ig_path,
                        occ_img_path=occ_path,
                        bb_img_path=bb_path,
                        hr_img_path=hr_path,
                        cam_img_path=cam_path,
                        cf_img_path=None, # Ignoring counterfactual here for speed
                        narratives=narratives,
                        similar_cases=[],
                        output_pdf_path=target_pdf,
                        disease_name=tgt,
                        include_technical=True
                    )
                    
                    print(f"      [V] Generated {label_name} Report -> {target_pdf}")
                    
                    # Safe Tidy
                    for p in [orig_path, gc_path, ig_path, occ_path, bb_path, hr_path, cam_path]:
                        if p and os.path.exists(p): os.remove(p)
                        
                    # Memory clear
                    del input_t, meta_t, gradcam_attr, ig_attr, occ_attr, explainer, wrapped_model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"      [X] Generation Failed for {label_name}: {e}")
                    traceback.print_exc()

if __name__ == "__main__":
    main()
