import os
import uuid
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException
from fastapi.responses import FileResponse
from torchvision.models import swin_v2_t
from torchvision import transforms
from PIL import Image

import sys
import numpy as np
# Allow importing from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explain.methods import ExplainabilityMethods
from explain.visualize import plot_attributions, generate_annotated_image, generate_high_res_annotation, generate_clinical_summary
from explain.vlm_synthesizer import init_vlm, synthesize_comprehensive_report
try:
    from explain.vlm_synthesizer import synthesize_clinical_only_report
except ImportError:
    synthesize_clinical_only_report = None
from api.pdf_generator import generate_clinical_pdf

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Multi-Domain Clinical Explainability API",
    description="Endpoint for producing PDF explainability reports from medical images using Swin Transformer V2 models across chest X-ray, dermoscopy, fundus, and brain MRI modalities."
)

# Allow mobile app connections from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registries for hot-swapping multiple disease models dynamically
MODELS_REGISTRY = {}
EXPLAINER_REGISTRY = {}
RETRIEVER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model_and_explainer(disease: str):
    """Dynamically loads and caches the Swin Transformer optimized for a specific disease."""
    if disease in MODELS_REGISTRY:
        return MODELS_REGISTRY[disease], EXPLAINER_REGISTRY[disease]
        
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    experiments_dir = project_root / 'experiments'
    
    # Recursively find the model file for the requested disease
    model_files = list(experiments_dir.rglob(f'{disease}_model.pth'))
    
    if not model_files:
        raise ValueError(f"Model weights for disease '{disease}' not found in {experiments_dir}.")
        
    model_path = str(model_files[0])
    print(f"Loading {disease} model from {model_path}")
    
    # Reconstruct the exact MultiModalFusion architecture used during training
    from torchvision.models import swin_v2_s
    
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
            if self.meta_dim > 0:
                if metadata is None:
                    metadata = torch.zeros(image.shape[0], self.meta_dim, device=image.device)
                fused_features = torch.cat((v_features, metadata), dim=1)
                return self.fusion_mlp(fused_features)
            return self.fusion_mlp(v_features)
    
    # Try loading with MultiModalFusion wrapper first
    base_model = swin_v2_s(weights=None)
    vision_dim = base_model.head.in_features  # 768 for swin_v2_s
    
    # Probe the state dict to determine metadata dimension
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    meta_dim = 0
    for key in state_dict:
        if 'fusion_mlp.0.weight' in key:
            # First linear layer input size = vision_dim + meta_dim
            total_input = state_dict[key].shape[1]
            meta_dim = total_input - vision_dim
            break
    
    if meta_dim < 0:
        meta_dim = 0
    
    print(f"Detected meta_dim={meta_dim} from saved weights.")
    model = MultiModalFusion(vision_model=base_model, vision_dim=vision_dim, meta_dim=meta_dim)
    
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded MultiModalFusion weights.")
    except Exception:
        # Fallback: try bare Swin model for legacy weights
        from torchvision.models import swin_v2_t
        model = swin_v2_t(weights=None)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 1)
        model.load_state_dict(state_dict)
        print("Loaded legacy Swin-T bare weights.")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Wrap for Captum explainability (exposes .features for SAS/GradCAM hooks)
    class PureVisionWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        @property
        def features(self):
            if hasattr(self.model, 'vision_model'):
                return self.model.vision_model.features
            return self.model.features
        def forward(self, img):
            return self.model(img)
    
    wrapped = PureVisionWrapper(model).to(DEVICE)
    wrapped.eval()
    explainer = ExplainabilityMethods(wrapped, device=DEVICE)
    
    MODELS_REGISTRY[disease] = model
    EXPLAINER_REGISTRY[disease] = explainer
    
    return model, explainer

def load_environment():
    """Initializes RAG database and template engine on startup."""
    global RETRIEVER
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Initialize RAG Database
    from explain.retrieval import RAGImageRetriever
    try:
        index_path = os.path.join(project_root, 'experiments', 'rag', 'index.pt')
        meta_path = os.path.join(project_root, 'experiments', 'rag', 'meta.json')
        # We need A model for the retriever to embed queries. We'll use the generic Pneumonia one
        base_model, _ = get_model_and_explainer("Pneumonia")
        RETRIEVER = RAGImageRetriever(base_model, index_path=index_path, meta_path=meta_path, device=DEVICE)
    except Exception as e:
        print(f"Warning: RAG index not built yet or failed to load. Error: {e}")
    
    # Template engine (no-op, no heavy model loading)
    init_vlm(DEVICE)

@app.on_event("startup")
async def startup_event():
    load_environment()

@app.get("/health", tags=["System"])
async def health():
    """Health check for mobile app connectivity."""
    return {"status": "ok", "models_loaded": len(MODELS_REGISTRY), "device": str(DEVICE)}

@app.post("/predict", tags=["Mobile Inference"])
async def predict(file: UploadFile = File(...), disease: str = Form("Pneumonia")):
    """Lightweight JSON prediction for mobile app. Returns probability without PDF/explainability overhead."""
    import torchvision.transforms.functional as TF
    
    try:
        model, _ = get_model_and_explainer(disease)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Read and preprocess image
    img_bytes = await file.read()
    from io import BytesIO
    img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
    input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    
    # TTA: original + hflip (fast, 2-pass)
    with torch.no_grad():
        out1 = model(input_tensor).squeeze()
        if out1.dim() == 0: out1 = out1.unsqueeze(0)
        p1 = torch.sigmoid(out1).item()
        
        img_flip = TF.hflip(img_pil)
        tensor_flip = TRANSFORM(img_flip).unsqueeze(0).to(DEVICE)
        out2 = model(tensor_flip).squeeze()
        if out2.dim() == 0: out2 = out2.unsqueeze(0)
        p2 = torch.sigmoid(out2).item()
    
    prob = (p1 + p2) / 2.0
    is_positive = prob > 0.5
    
    # Confidence level
    if prob > 0.85 or prob < 0.15:
        confidence = "HIGH"
    elif prob > 0.65 or prob < 0.35:
        confidence = "MODERATE"
    else:
        confidence = "LOW"
    
    return {
        "disease": disease,
        "probability": round(prob, 4),
        "prediction": "POSITIVE" if is_positive else "NEGATIVE",
        "confidence": confidence,
        "tta_passes": 2,
    }

@app.post("/predict/batch", tags=["Mobile Inference"])
async def predict_batch(file: UploadFile = File(...), diseases: str = Form("Pneumonia,Atelectasis")):
    """Batch prediction for multiple diseases on a single image."""
    import torchvision.transforms.functional as TF
    
    img_bytes = await file.read()
    from io import BytesIO
    img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
    input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    flip_tensor = TRANSFORM(TF.hflip(img_pil)).unsqueeze(0).to(DEVICE)
    
    disease_list = [d.strip() for d in diseases.split(",")]
    results = []
    
    for disease in disease_list:
        try:
            model, _ = get_model_and_explainer(disease)
            with torch.no_grad():
                out1 = model(input_tensor).squeeze()
                if out1.dim() == 0: out1 = out1.unsqueeze(0)
                p1 = torch.sigmoid(out1).item()
                
                out2 = model(flip_tensor).squeeze()
                if out2.dim() == 0: out2 = out2.unsqueeze(0)
                p2 = torch.sigmoid(out2).item()
            
            prob = (p1 + p2) / 2.0
            results.append({
                "disease": disease,
                "probability": round(prob, 4),
                "prediction": "POSITIVE" if prob > 0.5 else "NEGATIVE",
            })
        except Exception as e:
            results.append({"disease": disease, "error": str(e)})
    
    return {"results": results}

def cleanup_files(*filepaths):
    """Background task to tidy up the temp directories after the request finishes."""
    for path in filepaths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

@app.post("/analyze/clinical", tags=["Clinical Report"])
async def analyze_clinical(background_tasks: BackgroundTasks, file: UploadFile = File(...), disease: str = Form("Pneumonia")):
    """Generates an ultra-fast 1-page clinical output without any heatmaps or heavy AI inference."""
    return await _process_analysis(background_tasks, file, disease, include_technical=False, report_mode="professional")

@app.post("/analyze/comprehensive", tags=["Comprehensive Report"])
async def analyze_comprehensive(background_tasks: BackgroundTasks, file: UploadFile = File(...), disease: str = Form("Pneumonia")):
    """Generates the full professional report including SAS, IG, Occlusion, and RAG."""
    return await _process_analysis(background_tasks, file, disease, include_technical=True, report_mode="professional")

@app.post("/analyze/public", tags=["Public Health Screening"])
async def analyze_public(background_tasks: BackgroundTasks, file: UploadFile = File(...), disease: str = Form("Pneumonia")):
    """Generates a simplified health screening report for general public users.
    Includes disease info, visual comparison guide, and plain-language findings.
    Excludes detailed explainability heatmaps and technical AI metrics."""
    return await _process_analysis(background_tasks, file, disease, include_technical=False, report_mode="public")

@app.post("/analyze/professional", tags=["Professional Report"])
async def analyze_professional(background_tasks: BackgroundTasks, file: UploadFile = File(...), disease: str = Form("Pneumonia")):
    """Generates a full professional clinical report with comprehensive explainability.
    Includes SAS, IG, Occlusion heatmaps, RAG retrieval, and clinician verdict fields."""
    return await _process_analysis(background_tasks, file, disease, include_technical=True, report_mode="professional")

async def _process_analysis(
    background_tasks: BackgroundTasks, 
    file: UploadFile,
    disease: str,
    include_technical: bool,
    report_mode: str = "professional",
):
    try:
        model, explainer = get_model_and_explainer(disease)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # 1. Setup Request Workspaces
    req_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.abspath("experiments/api_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    input_img_path = os.path.join(temp_dir, f"{req_id}_input.png")
    
    # Define paths for the 3 individual heatmap artifacts
    # We will generate them by calling plot_attributions three separate times for the single-method overlays.
    gradcam_img_path = os.path.join(temp_dir, f"{req_id}_gradcam.png")
    ig_img_path = os.path.join(temp_dir, f"{req_id}_ig.png")
    occ_img_path = os.path.join(temp_dir, f"{req_id}_occlusion.png")
    bb_img_path = os.path.join(temp_dir, f"{req_id}_bounding_box.png")
    hr_img_path = os.path.join(temp_dir, f"{req_id}_high_res.png")
    cf_img_path = os.path.join(temp_dir, f"{req_id}_counterfactual.png")
    cam_img_path = os.path.join(temp_dir, f"{req_id}_cam.png")
    pdf_report_path = os.path.join(temp_dir, f"{req_id}_Clinical_Report.pdf")
    
    # 2. Save Uploaded Image
    with open(input_img_path, "wb") as f:
        f.write(await file.read())
        
    # 3. Process Image
    # Convert exactly as the standard Dataset pipeline does: 1 channel to 3 identical RGB channels
    img_pil = Image.open(input_img_path).convert('RGB')
    
    # Keep a reference to the 256x256 raw image for the PDF overlay
    img_resized = img_pil.resize((256, 256))
    img_resized.save(input_img_path) 
    
    input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True # Required for Grad-CAM
    
    # 4. Model Inference (Test-Time Augmentation for Extreme Confidence)
    import torchvision.transforms.functional as TF
    
    tta_probs = []
    with torch.no_grad():
        # Pass 1: Original
        out1 = model(input_tensor).squeeze()
        if out1.dim() == 0: out1 = out1.unsqueeze(0)
        tta_probs.append(torch.sigmoid(out1 / 0.50).item())
        
        # Pass 2: Horizontal Flip
        img_flip = TF.hflip(img_pil)
        tensor_flip = TRANSFORM(img_flip).unsqueeze(0).to(DEVICE)
        out2 = model(tensor_flip).squeeze()
        if out2.dim() == 0: out2 = out2.unsqueeze(0)
        tta_probs.append(torch.sigmoid(out2 / 0.50).item())
        
        # Pass 3: Rotation +10 deg
        img_rot1 = TF.rotate(img_pil, 10)
        tensor_rot1 = TRANSFORM(img_rot1).unsqueeze(0).to(DEVICE)
        out3 = model(tensor_rot1).squeeze()
        if out3.dim() == 0: out3 = out3.unsqueeze(0)
        tta_probs.append(torch.sigmoid(out3 / 0.50).item())
        
        # Pass 4: Rotation -10 deg
        img_rot2 = TF.rotate(img_pil, -10)
        tensor_rot2 = TRANSFORM(img_rot2).unsqueeze(0).to(DEVICE)
        out4 = model(tensor_rot2).squeeze()
        if out4.dim() == 0: out4 = out4.unsqueeze(0)
        tta_probs.append(torch.sigmoid(out4 / 0.50).item())
        
        # Pass 5: Color Jitter (Brightness/Contrast shift)
        jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        img_jit = jitter(img_pil)
        tensor_jit = TRANSFORM(img_jit).unsqueeze(0).to(DEVICE)
        out5 = model(tensor_jit).squeeze()
        if out5.dim() == 0: out5 = out5.unsqueeze(0)
        tta_probs.append(torch.sigmoid(out5 / 0.50).item())
        
    prob = sum(tta_probs) / 5.0
        
    # 4.1 Monte Carlo (MC) Dropout for Uncertainty Quantification
    def enable_dropout(m):
        """Force dropout/stochastic layers to be active for MC inference"""
        if type(m) == nn.Dropout or "Drop" in str(type(m)):
            m.train()
            
    model.apply(enable_dropout)
    mc_probs = []
    with torch.no_grad():
        for _ in range(30): # 30 stochastic forward passes
            mc_out = model(input_tensor).squeeze()
            if mc_out.dim() == 0: mc_out = mc_out.unsqueeze(0)
            mc_probs.append(torch.sigmoid(mc_out).item())
            
    mc_probs = np.array(mc_probs)
    # The standard deviation of the stochastic predictions yields the uncertainty.
    # Higher deviation = model is guessing/unstable on this specific input.
    uncertainty_score = float(mc_probs.std())
    
    # Restore rigid evaluation mode
    model.eval()
        
    # 5. Generate Explanations
    if include_technical:
        # Semantic Attention Synthesis (Also use SAS matrix to draw the clinical bounding boxes)
        gradcam_attr = explainer.generate_sas(input_tensor)
        
        # We use a raw, un-normalized tensor solely for clean visual plotting over the original image
        unbatched_vis = transforms.ToTensor()(img_resized).to(DEVICE)
        
        plot_attributions(unbatched_vis, {"Semantic Attention Synthesis": gradcam_attr}, gradcam_img_path, "Unknown", prob)
        
        # Integrated Gradients
        attr_ig = explainer.generate_integrated_gradients(input_tensor, target_class=0, n_steps=25)
        plot_attributions(unbatched_vis, {"Integrated Gradients": attr_ig}, ig_img_path, "Unknown", prob)
        
        # Occlusion
        attr_occ = explainer.generate_occlusion(input_tensor, target_class=0, sliding_window_shapes=(3, 15, 15), strides=(3, 8, 8))
        plot_attributions(unbatched_vis, {"Occlusion": attr_occ}, occ_img_path, "Unknown", prob)
        
        # Bounding Box: Multi-method consensus fusion (SAS primary + IG + Occlusion)
        roc_area_pct, spatial_regions = generate_annotated_image(
            unbatched_vis, gradcam_attr.detach(), bb_img_path, threshold=0.7,
            secondary_attrs=[attr_ig.detach(), attr_occ.detach()]
        )
        
        # Generate the ultra sharp anatomical contours from the IG heatmap
        generate_high_res_annotation(unbatched_vis, attr_ig.detach(), hr_img_path, threshold=0.88)
        
        # CAM (Class Activation Mapping) — spatial features before pooling
        try:
            import cv2 as cv2_cam
            vm = model.vision_model if hasattr(model, 'vision_model') else model
            with torch.no_grad():
                spatial = vm.features(input_tensor)
                spatial = vm.norm(spatial)
                spatial = vm.permute(spatial)  # [1, 768, 8, 8]
            w_key = None
            for key in model.state_dict():
                if 'fusion_mlp.0.weight' in key:
                    w_key = key; break
            if w_key is None:
                for key in model.state_dict():
                    if 'fusion_mlp.1.weight' in key:
                        w_key = key; break
            if w_key:
                weights = model.state_dict()[w_key][:, :768]
                channel_w = weights.sum(dim=0)
                cam_map = torch.zeros(8, 8, device=DEVICE)
                for c in range(768):
                    cam_map += channel_w[c] * spatial[0, c]
                cam_map = torch.relu(cam_map)
                if cam_map.max() > 0:
                    cam_map = cam_map / cam_map.max()
                cam_np = cam_map.cpu().numpy()
                cam_resized = cv2_cam.resize(cam_np, (256, 256), interpolation=cv2_cam.INTER_LINEAR)
                cam_colored = cv2_cam.applyColorMap((cam_resized * 255).astype(np.uint8), cv2_cam.COLORMAP_JET)
                orig_bgr = cv2_cam.cvtColor((np.transpose(unbatched_vis.cpu().numpy(), (1,2,0)) * 255).clip(0,255).astype(np.uint8), cv2_cam.COLOR_RGB2BGR)
                cam_overlay = cv2_cam.addWeighted(orig_bgr, 0.6, cam_colored, 0.4, 0)
                cv2_cam.imwrite(cam_img_path, cam_overlay)
            else:
                cam_img_path = None
        except Exception as cam_err:
            print(f"CAM generation failed: {cam_err}")
            cam_img_path = None

        # Generative Counterfactual Inpainting
        try:
            from explain.counterfactual import generate_counterfactual
            generate_counterfactual(input_img_path, gradcam_attr, cf_img_path, threshold=0.7)
        except Exception as e:
            print(f"Counterfactual synthesis failed: {e}")
            cf_img_path = None
    else:
        # Fast endpoint bypasses everything
        roc_area_pct = 0.0
        spatial_regions = []
        cam_img_path = None
    
    # 6. Generate Human-Friendly Text Summary via VLM
    # Fallback heuristic summary
    summary_text = generate_clinical_summary(prob, roc_area_pct, disease_name=disease)
    narratives = {
        "overall": summary_text,
        "gradcam": "No detailed analysis generated.", # This key remains "gradcam" for consistency with VLM prompt structure
        "ig": "No detailed analysis generated.",
        "occlusion": "No detailed analysis generated."
    }
    
    try:
        if include_technical:
            narratives = synthesize_comprehensive_report(
                probability=prob,
                spatial_regions=spatial_regions,
                disease_name=disease,
                area_pct=roc_area_pct,
                uncertainty=uncertainty_score
            )
        else:
            narratives = synthesize_clinical_only_report(
                probability=prob,
                spatial_regions=spatial_regions,
                disease_name=disease,
                uncertainty=uncertainty_score
            )
            
    except Exception as e:
        print(f"Local Multi-Modal Synthesis Failed. Falling back to rule-based heuristic. Error: {e}")

    # 6.5 RAG Similarity Search
    similar_cases = []
    if include_technical and RETRIEVER is not None:
        try:
            similar_cases = RETRIEVER.retrieve_similar(input_tensor, k=3)
        except Exception as e:
            print(f"RAG search failed: {e}")

    # 7. Generate PDF Report (dual-mode aware)
    # Look for a reference positive image for public mode visual comparison
    reference_positive_img_path = None
    if report_mode == "public":
        from pathlib import Path
        ref_dir = Path(__file__).parent.parent.parent / 'assets' / 'reference_positives'
        ref_candidates = list(ref_dir.glob(f'{disease}.*')) if ref_dir.exists() else []
        if ref_candidates:
            reference_positive_img_path = str(ref_candidates[0])

    generate_clinical_pdf(
        patient_id=f"ANON-{req_id.upper()}",
        prediction_prob=prob,
        uncertainty_score=uncertainty_score,
        original_img_path=input_img_path,
        gradcam_img_path=gradcam_img_path,
        ig_img_path=ig_img_path,
        occ_img_path=occ_img_path,
        bb_img_path=bb_img_path,
        hr_img_path=hr_img_path,
        cf_img_path=cf_img_path,
        cam_img_path=cam_img_path,
        narratives=narratives,
        similar_cases=similar_cases,
        output_pdf_path=pdf_report_path,
        disease_name=disease,
        include_technical=include_technical,
        report_mode=report_mode,
        reference_positive_img_path=reference_positive_img_path,
    )
    
    # 8. Cleanup the image temp files (keep PDF until stream finishes)
    background_tasks.add_task(cleanup_files, input_img_path, gradcam_img_path, ig_img_path, occ_img_path, bb_img_path, hr_img_path, cf_img_path, cam_img_path, pdf_report_path)
    
    # 8. Return PDF Download Response
    return FileResponse(
        path=pdf_report_path, 
        media_type="application/pdf", 
        filename=f"Patient_Report_{req_id}.pdf"
    )
    
if __name__ == "__main__":
    import uvicorn
    # Make sure we run from the project root L:\MTP
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
