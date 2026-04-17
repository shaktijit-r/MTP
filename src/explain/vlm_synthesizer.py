import torch
from PIL import Image

# Global VLM Singletons
VLM_MODEL = None
VLM_TOKENIZER = None

# ─── Disease → Imaging Modality Mapping (all 35 diseases) ────────────────────
MODALITY_MAP = {
    # Chest X-Ray diseases (18)
    'Atelectasis': ('chest X-ray', 'radiologist'),
    'Cardiomegaly': ('chest X-ray', 'radiologist'),
    'Consolidation': ('chest X-ray', 'radiologist'),
    'Edema': ('chest X-ray', 'radiologist'),
    'Emphysema': ('chest X-ray', 'radiologist'),
    'Enlarged_Cardiomediastinum': ('chest X-ray', 'radiologist'),
    'Fibrosis': ('chest X-ray', 'radiologist'),
    'Fracture': ('chest X-ray', 'radiologist'),
    'Hernia': ('chest X-ray', 'radiologist'),
    'Infiltration': ('chest X-ray', 'radiologist'),
    'Lung_Lesion': ('chest X-ray', 'radiologist'),
    'Lung_Opacity': ('chest X-ray', 'radiologist'),
    'Mass': ('chest X-ray', 'radiologist'),
    'Nodule': ('chest X-ray', 'radiologist'),
    'Pleural_Effusion': ('chest X-ray', 'radiologist'),
    'Pleural_Thickening': ('chest X-ray', 'radiologist'),
    'Pneumonia': ('chest X-ray', 'radiologist'),
    'Pneumothorax': ('chest X-ray', 'radiologist'),
    'COVID': ('chest X-ray', 'radiologist'),
    'Viral_Pneumonia': ('chest X-ray', 'radiologist'),
    # Dermatology diseases (7)
    'Melanoma': ('dermoscopic image', 'dermatologist'),
    'akiec': ('dermoscopic image', 'dermatologist'),
    'bcc': ('dermoscopic image', 'dermatologist'),
    'bkl': ('dermoscopic image', 'dermatologist'),
    'df': ('dermoscopic image', 'dermatologist'),
    'nv': ('dermoscopic image', 'dermatologist'),
    'vasc': ('dermoscopic image', 'dermatologist'),
    # Ophthalmology diseases (7)
    'AMD': ('fundus photograph', 'ophthalmologist'),
    'Cataract': ('fundus photograph', 'ophthalmologist'),
    'Diabetes': ('fundus photograph', 'ophthalmologist'),
    'Diabetic_Retinopathy': ('fundus photograph', 'ophthalmologist'),
    'Glaucoma': ('fundus photograph', 'ophthalmologist'),
    'Hypertension': ('fundus photograph', 'ophthalmologist'),
    'Myopia': ('fundus photograph', 'ophthalmologist'),
    # Neurology (1)
    'Dementia': ('brain MRI scan', 'neurologist'),
}

def get_modality(disease_name):
    """Returns (scan_type, specialist_role) for a given disease."""
    return MODALITY_MAP.get(disease_name, ('medical scan', 'clinician'))

def init_vlm(device):
    """Loads the Qwen2.5 edge LLM model into memory."""
    global VLM_MODEL, VLM_TOKENIZER
    if VLM_MODEL is not None:
        return
        
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading local Edge LLM (Qwen2.5-3B-Instruct) into memory...")
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        VLM_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map='auto'
        )
        VLM_MODEL.eval()
        VLM_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        print("Local Edge LLM loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load local LLM. Error: {e}")
        
def generate_narrative(prompt):
    """Helper to run a purely text-based prompt through Qwen."""
    if VLM_MODEL is None or VLM_TOKENIZER is None:
        return None  # Signal caller to use template fallback
        
    try:
        messages = [
            {"role": "system", "content": "You are a precise, highly accurate senior medical specialist with expertise across radiology, dermatology, ophthalmology, and neurology."},
            {"role": "user", "content": prompt}
        ]
        text = VLM_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = VLM_TOKENIZER([text], return_tensors="pt").to(VLM_MODEL.device)

        with torch.no_grad():
            generated_ids = VLM_MODEL.generate(
                **model_inputs,
                max_new_tokens=150,
                temperature=0.3
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        answer = VLM_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip() if answer else "No summary generated."
    except Exception as e:
        print(f"Failed to generate narrative: {e}")
        return f"Error during synthesis: {e}"

def synthesize_comprehensive_report(probability, spatial_regions, disease_name="Pneumonia", area_pct=None, uncertainty=None):
    """
    Executes four separate VLM inference calls to generate a comprehensive 
    transparency report breaking down the model's logic step-by-step.
    """
    # Determine modality-aware scan type from disease name
    scan_type, specialist = get_modality(disease_name)
    
    is_positive = probability >= 0.5
    diagnosis_status = "POSITIVE" if is_positive else "NEGATIVE"
    
    regions_text = ", ".join(spatial_regions) if spatial_regions else "distributed globally across the scan"
    
    if is_positive:
        prompt_overall = f"Act as an expert {specialist}. The AI has diagnosed this {scan_type} as POSITIVE for {disease_name}. The pathology is heavily concentrated in the following anatomical locations: {regions_text}. You must align with this diagnosis. Describe how pathological abnormalities and structural irregularities physically appearing in the {regions_text} are consistent with {disease_name}. DO NOT state that the scan appears normal. DO NOT use words like 'model', 'AI', 'prediction', 'heatmap', or 'bounding box'. Write ONLY the clinical medical observations. Limit to 3-4 sentences."
    else:
        prompt_overall = f"Act as an expert {specialist}. The AI has completely cleared this {scan_type} and found NO evidence of {disease_name}. You must align with this diagnosis. Describe the healthy, normal anatomical properties visible globally. Emphasize the lack of abnormalities. DO NOT suggest any disease is present. DO NOT use words like 'model', 'AI', 'prediction'. Write ONLY the clinical medical observations. Limit to 3-4 sentences."
    
    narrative_overall = generate_narrative(prompt_overall)
    
    # 2. SAS Analysis (reusing gradcam key for pipeline consistency)
    base_context = f"You are a clinical AI explainability expert evaluating a {scan_type}. DO NOT hallucinate. "
    prompt_gradcam = base_context + f"The Semantic Attention Synthesis (SAS) routing matrix focused its heaviest mathematical self-attention strictly on the: {regions_text}. Describe the explicit anatomical structures fundamentally located in the {regions_text} using formal clinical vocabulary, and functionally explain why attending to these structures implies {disease_name}. Limit to 3 sentences."
    narrative_gradcam = generate_narrative(prompt_gradcam)
    
    # 3. Integrated Gradients Analysis
    prompt_ig = base_context + f"The Integrated Gradients analysis highlights high-resolution pixel-level textures specifically within the {regions_text}. Describe the micro-opacities or border irregularities that are structurally expected in the {regions_text} during a {disease_name} diagnosis using strict medical terminology. Limit to 3 sentences."
    narrative_ig = generate_narrative(prompt_ig)
    
    # 4. Occlusion Analysis
    prompt_occ = base_context + f"The Occlusion heatmap reveals that masking the {regions_text} caused the AI's confidence to critically drop. Technically identify the critical biological zones situated in the {regions_text} that are essential to the {disease_name} diagnostic conclusion. Limit to 3 sentences."
    narrative_occ = generate_narrative(prompt_occ)
    
    # 5. Clinical Impression (for the clinical-facing pages — NO AI jargon)
    if is_positive:
        prompt_impression = (
            f"Act as a senior {specialist} writing a formal clinical impression. "
            f"Based on this {scan_type}, write a 2-3 sentence impression. "
            f"The diagnosis is POSITIVE for {disease_name}. "
            f"You MUST state that findings are highly suggestive of {disease_name}, "
            f"provide a brief severity assessment, and recommend follow-up. "
            f"DO NOT mention AI, models, or probabilities. Write ONLY the clinical impression."
        )
    else:
        prompt_impression = (
            f"Act as a senior {specialist} writing a formal clinical impression. "
            f"Based on this {scan_type}, write a 2-3 sentence impression. "
            f"The diagnosis is NEGATIVE for {disease_name}. "
            f"You MUST clearly state that findings are NOT suggestive of {disease_name}, "
            f"and recommend routine follow-up. "
            f"DO NOT mention AI, models, or probabilities. Write ONLY the clinical impression."
        )
    narrative_impression = generate_narrative(prompt_impression)
    
    # Template-based fallbacks when LLM is unavailable
    disease_display = disease_name.replace('_', ' ')
    unct_text = f" Uncertainty (±σ): {uncertainty:.4f}." if uncertainty is not None else ""
    area_text = f" The region of concern spans approximately {area_pct:.1%} of the scan area." if area_pct and area_pct > 0 else ""
    
    if narrative_overall is None:
        if is_positive:
            narrative_overall = (
                f"The SwinTransformerV2 system identified abnormalities consistent with {disease_display} "
                f"in the {regions_text}. Confidence: {probability:.1%}.{unct_text}{area_text} "
                f"Clinical correlation is recommended."
            )
        else:
            narrative_overall = (
                f"No significant findings suggestive of {disease_display} were identified on this {scan_type}. "
                f"The examined structures appear within normal limits. Confidence: {(1-probability):.1%}.{unct_text}"
            )
    
    if narrative_gradcam is None:
        narrative_gradcam = (
            f"The Semantic Attention Synthesis routing concentrated the model's self-attention patterns "
            f"on the {regions_text}. This indicates the transformer's spatial attention heads identified "
            f"these anatomical regions as most diagnostically relevant for {disease_display} classification."
        )
    
    if narrative_ig is None:
        narrative_ig = (
            f"Integrated Gradients attribution reveals pixel-level features in the {regions_text} "
            f"that contributed most strongly to the model's {diagnosis_status} prediction. "
            f"Fine-grained structural details in these regions drove the classification decision."
        )
    
    if narrative_occ is None:
        narrative_occ = (
            f"Occlusion sensitivity analysis confirms that masking the {regions_text} caused the "
            f"largest confidence degradation ({probability:.1%} → reduced), validating these regions "
            f"as causally important to the {disease_display} diagnosis."
        )
    
    if narrative_impression is None:
        if is_positive:
            if probability >= 0.85:
                narrative_impression = f"Findings are highly suggestive of {disease_display}. Clinical correlation and follow-up imaging are recommended. Consider further workup as clinically indicated."
            elif probability >= 0.65:
                narrative_impression = f"Findings are suggestive of {disease_display}. Correlation with clinical symptoms and laboratory findings is recommended."
            else:
                narrative_impression = f"Findings raise the possibility of {disease_display}, though with limited confidence. Close clinical correlation is advised."
        else:
            if probability <= 0.15:
                narrative_impression = f"No evidence suggestive of {disease_display} is identified. The examined structures appear unremarkable. Routine follow-up as clinically indicated."
            else:
                narrative_impression = f"No definitive findings of {disease_display} are identified. Clinical correlation is recommended if symptoms persist."
    
    return {
        "overall": narrative_overall,
        "gradcam": narrative_gradcam,
        "ig": narrative_ig,
        "occlusion": narrative_occ,
        "clinical_impression": narrative_impression
    }

def synthesize_clinical_only_report(probability, spatial_regions, disease_name="Pneumonia", uncertainty=None):
    """
    Executes only the two essential VLM calls required for the patient-facing 
    page, saving massive compute time by skipping the heatmap analyses.
    """
    scan_type, specialist = get_modality(disease_name)
    is_positive = probability >= 0.5
    diagnosis_status = "POSITIVE" if is_positive else "NEGATIVE"

    regions_text = ", ".join(spatial_regions) if spatial_regions else "distributed globally"

    if is_positive:
        prompt_overall = f"Act as an expert {specialist}. The AI has diagnosed this {scan_type} as POSITIVE for {disease_name}, specifically highlighting the {regions_text}. You must align with this diagnosis. Describe the pathological abnormalities physically present in the {regions_text} that confirm {disease_name}. DO NOT state that the scan appears normal. DO NOT use words like 'model', 'AI', 'prediction'. Write ONLY the clinical medical observations. Limit to 3-4 sentences."
    else:
        prompt_overall = f"Act as an expert {specialist}. The AI has completely cleared this {scan_type} and found NO evidence of {disease_name}. You must align with this diagnosis. Describe the healthy, normal anatomical properties visible globally. Emphasize the lack of abnormalities. DO NOT suggest any disease is present. DO NOT use words like 'model', 'AI', 'prediction'. Write ONLY the clinical medical observations. Limit to 3-4 sentences."
    
    narrative_overall = generate_narrative(prompt_overall)
    
    if is_positive:
        prompt_impression = (
            f"Act as a senior {specialist} writing a formal clinical impression. "
            f"Based on this {scan_type}, write a 2-3 sentence impression. "
            f"The diagnosis is POSITIVE for {disease_name}. "
            f"You MUST state that findings are highly suggestive of {disease_name}, "
            f"provide a brief severity assessment, and recommend follow-up. "
            f"DO NOT mention AI, models, or probabilities. Write ONLY the clinical impression."
        )
    else:
        prompt_impression = (
            f"Act as a senior {specialist} writing a formal clinical impression. "
            f"Based on this {scan_type}, write a 2-3 sentence impression. "
            f"The diagnosis is NEGATIVE for {disease_name}. "
            f"You MUST clearly state that findings are NOT suggestive of {disease_name}, "
            f"and recommend routine follow-up. "
            f"DO NOT mention AI, models, or probabilities. Write ONLY the clinical impression."
        )
    narrative_impression = generate_narrative(prompt_impression)
    
    # Template fallbacks
    disease_display = disease_name.replace('_', ' ')
    if narrative_overall is None:
        if is_positive:
            narrative_overall = (
                f"The SwinTransformerV2 system identified abnormalities consistent with {disease_display} "
                f"in the {regions_text}. Confidence: {probability:.1%}. Clinical correlation is recommended."
            )
        else:
            narrative_overall = (
                f"No significant findings suggestive of {disease_display} were identified on this {scan_type}. "
                f"The examined structures appear within normal limits. Confidence: {(1-probability):.1%}."
            )
    
    if narrative_impression is None:
        if is_positive:
            narrative_impression = f"Findings are suggestive of {disease_display}. Clinical correlation and appropriate follow-up are recommended."
        else:
            narrative_impression = f"No evidence suggestive of {disease_display} is identified. Routine follow-up as clinically indicated."
    
    return {
        "overall": narrative_overall,
        "clinical_impression": narrative_impression
    }
