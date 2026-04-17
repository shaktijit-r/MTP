import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.utils import ImageReader

# ─── Theme Colors ─────────────────────────────────────────────────────────────
THEME = {
    "professional": {
        "primary": "#1565C0",        # Clinical blue
        "primary_light": "#E3F2FD",
        "header_bg": "#1a1a2e",
        "section_color": "#1a1a2e",
        "banner_pos_bg": "#FFF0F0",
        "banner_pos_border": "#CC0000",
        "banner_pos_text": "#CC0000",
        "banner_neg_bg": "#F0FFF0",
        "banner_neg_border": "#006600",
        "banner_neg_text": "#006600",
        "banner_unc_bg": "#FFFBF0",
        "banner_unc_border": "#CC9900",
        "banner_unc_text": "#CC9900",
        "table_header_bg": "#D9E2F3",
        "mode_label": "CLINICAL MODE — For Healthcare Professionals",
    },
    "public": {
        "primary": "#00897B",        # Warm teal-green
        "primary_light": "#E0F2F1",
        "header_bg": "#004D40",
        "section_color": "#004D40",
        "banner_pos_bg": "#FFF3E0",
        "banner_pos_border": "#E65100",
        "banner_pos_text": "#BF360C",
        "banner_neg_bg": "#E8F5E9",
        "banner_neg_border": "#2E7D32",
        "banner_neg_text": "#1B5E20",
        "banner_unc_bg": "#FFF8E1",
        "banner_unc_border": "#F9A825",
        "banner_unc_text": "#F57F17",
        "table_header_bg": "#B2DFDB",
        "mode_label": "HEALTH SCREENING — General Public Report",
    }
}

# ─── Disease Encyclopedia (for General Public Mode) ───────────────────────────
DISEASE_INFO = {
    "Atelectasis": {
        "name": "Atelectasis (Collapsed Lung)",
        "what": "A condition where part or all of a lung collapses or does not inflate properly, reducing the amount of oxygen reaching the blood.",
        "symptoms": "Shortness of breath, rapid shallow breathing, cough, chest pain (in severe cases).",
        "risk_factors": "Recent surgery (especially abdominal/chest), prolonged bed rest, smoking, obesity, mucus plugs.",
        "severity": "Ranges from minor (small area) to serious (entire lung). Small atelectasis may resolve on its own.",
        "next_steps": "Consult a pulmonologist. Treatment may include breathing exercises, chest physiotherapy, or bronchoscopy.",
    },
    "Cardiomegaly": {
        "name": "Cardiomegaly (Enlarged Heart)",
        "what": "An enlarged heart visible on imaging, often a sign of an underlying condition like hypertension, valve disease, or cardiomyopathy.",
        "symptoms": "Shortness of breath, swelling in legs/ankles, fatigue, irregular heartbeat.",
        "risk_factors": "High blood pressure, coronary artery disease, valve disorders, family history of heart disease.",
        "severity": "Requires medical evaluation. An enlarged heart can lead to heart failure if the underlying cause is not treated.",
        "next_steps": "See a cardiologist for echocardiography and further cardiac workup.",
    },
    "Consolidation": {
        "name": "Lung Consolidation",
        "what": "A condition where lung tissue fills with fluid, pus, or inflammatory material instead of air, often due to infection.",
        "symptoms": "Cough (productive), fever, difficulty breathing, chest pain when breathing deeply.",
        "risk_factors": "Bacterial or viral infections, aspiration, immune deficiency.",
        "severity": "Can indicate pneumonia or other serious lung infection. Requires prompt medical attention.",
        "next_steps": "Consult a physician. Blood tests and sputum culture may be needed. Antibiotics are often prescribed.",
    },
    "Edema": {
        "name": "Pulmonary Edema (Fluid in Lungs)",
        "what": "Excess fluid accumulation in the lungs, making breathing difficult. Often related to heart problems.",
        "symptoms": "Severe shortness of breath, gasping for air (especially lying down), wheezing, pink frothy sputum.",
        "risk_factors": "Heart failure, kidney disease, high altitude, toxic inhalation.",
        "severity": "Can be life-threatening. Acute pulmonary edema is a medical emergency.",
        "next_steps": "Seek immediate medical care. Treatment focuses on the underlying cause (diuretics, oxygen, heart medication).",
    },
    "Pleural_Effusion": {
        "name": "Pleural Effusion (Fluid Around Lungs)",
        "what": "Abnormal buildup of fluid between the layers of tissue (pleura) that line the lungs and chest cavity.",
        "symptoms": "Chest pain, dry cough, difficulty breathing, reduced ability to exercise.",
        "risk_factors": "Heart failure, liver cirrhosis, pneumonia, cancer, pulmonary embolism.",
        "severity": "Small effusions may resolve on their own. Large effusions can compress the lung and require drainage.",
        "next_steps": "See a doctor for thoracentesis (fluid sampling) and underlying cause assessment.",
    },
    "Lung_Opacity": {
        "name": "Lung Opacity",
        "what": "An area on the X-ray that appears whiter than normal, indicating something other than air in the lung tissue (fluid, inflammation, tumor, or infection).",
        "symptoms": "Varies by cause — may include cough, fever, shortness of breath, or no symptoms at all.",
        "risk_factors": "Infections, smoking, occupational exposure, autoimmune conditions.",
        "severity": "Not a specific disease but a finding that needs further investigation to determine the cause.",
        "next_steps": "Follow up with a physician for CT scan or additional imaging to identify the underlying cause.",
    },
    "Pneumonia": {
        "name": "Pneumonia (Lung Infection)",
        "what": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.",
        "symptoms": "Cough with phlegm, fever/chills, shortness of breath, chest pain, fatigue, nausea.",
        "risk_factors": "Age (very young or elderly), weakened immune system, chronic lung disease, smoking, recent hospitalization.",
        "severity": "Can range from mild to life-threatening. Most dangerous for infants, older adults, and immunocompromised.",
        "next_steps": "Consult a doctor immediately. Treatment usually includes antibiotics, rest, and fluids.",
    },
    "Pneumothorax": {
        "name": "Pneumothorax (Collapsed Lung from Air Leak)",
        "what": "Air leaks into the space between the lung and chest wall, causing the lung to collapse partially or fully.",
        "symptoms": "Sudden sharp chest pain, shortness of breath, rapid heart rate.",
        "risk_factors": "Tall thin body type, smoking, lung disease, chest trauma, mechanical ventilation.",
        "severity": "Small pneumothorax may heal on its own. Large ones require emergency chest tube insertion.",
        "next_steps": "Seek emergency medical care if experiencing sudden chest pain with breathing difficulty.",
    },
    "Melanoma": {
        "name": "Melanoma (Skin Cancer)",
        "what": "The most serious type of skin cancer, developing in melanocyte cells that give skin its color.",
        "symptoms": "A new unusual growth or a change in an existing mole. Follow the ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving over time.",
        "risk_factors": "Excessive UV exposure, history of sunburns, fair skin, family history of melanoma, many moles.",
        "severity": "Highly treatable when caught early (Stage I: >95% survival). Advanced melanoma can spread to other organs.",
        "next_steps": "See a dermatologist immediately for biopsy. Early detection is critical for successful treatment.",
    },
    "Emphysema": {
        "name": "Emphysema",
        "what": "A chronic lung condition where the air sacs (alveoli) are damaged and enlarged, reducing the surface area for gas exchange.",
        "symptoms": "Chronic shortness of breath, persistent cough, wheezing, barrel-shaped chest.",
        "risk_factors": "Smoking (primary cause), long-term exposure to air pollution, alpha-1 antitrypsin deficiency.",
        "severity": "Irreversible lung damage. Progressive disease that worsens over time without management.",
        "next_steps": "Consult a pulmonologist. Quit smoking immediately. Treatment includes bronchodilators and pulmonary rehabilitation.",
    },
    "COVID": {
        "name": "COVID-19 Pneumonia",
        "what": "Lung infection caused by the SARS-CoV-2 virus, characterized by ground-glass opacities visible on imaging.",
        "symptoms": "Fever, cough, shortness of breath, fatigue, loss of taste/smell, body aches.",
        "risk_factors": "Unvaccinated status, advanced age, obesity, diabetes, cardiovascular disease.",
        "severity": "Ranges from asymptomatic to severe respiratory failure requiring ICU care.",
        "next_steps": "Isolate and contact a healthcare provider. Antiviral treatment may be indicated. Monitor oxygen levels.",
    },
    "Diabetic_Retinopathy": {
        "name": "Diabetic Retinopathy",
        "what": "Damage to the blood vessels in the retina caused by high blood sugar levels in diabetes patients.",
        "symptoms": "Often no symptoms in early stages. Later: blurred vision, floaters, dark areas in vision, vision loss.",
        "risk_factors": "Poorly controlled diabetes, long duration of diabetes, high blood pressure, high cholesterol.",
        "severity": "Leading cause of blindness in working-age adults. Irreversible vision loss if untreated.",
        "next_steps": "See an ophthalmologist for dilated eye exam. Control blood sugar, blood pressure, and cholesterol.",
    },
    "Glaucoma": {
        "name": "Glaucoma",
        "what": "A group of eye conditions that damage the optic nerve, usually due to abnormally high pressure inside the eye.",
        "symptoms": "Often no early symptoms ('silent thief of sight'). Late: peripheral vision loss, tunnel vision, eye pain.",
        "risk_factors": "Age >60, family history, elevated eye pressure, thin corneas, African/Hispanic ancestry.",
        "severity": "Second leading cause of blindness worldwide. Vision loss is irreversible but progression can be stopped.",
        "next_steps": "Regular eye exams are essential. Treatment includes eye drops, laser treatment, or surgery.",
    },
    "Dementia": {
        "name": "Dementia (Cognitive Decline)",
        "what": "A general term for a decline in cognitive function severe enough to interfere with daily life. Most common form is Alzheimer's disease.",
        "symptoms": "Memory loss, confusion, difficulty with language, impaired judgment, personality changes.",
        "risk_factors": "Age (strongest factor), family history, cardiovascular risk factors, head injuries, social isolation.",
        "severity": "Progressive and currently incurable, but early diagnosis enables management and planning.",
        "next_steps": "Consult a neurologist for cognitive testing and brain imaging. Support services are available.",
    },
}

# Fallback for diseases not in the dictionary
DEFAULT_DISEASE_INFO = {
    "name": "Medical Condition",
    "what": "A medical condition detected by AI analysis of the provided medical image.",
    "symptoms": "Varies depending on the specific condition. Consult a healthcare provider for personalized information.",
    "risk_factors": "Various factors may contribute. A medical professional can assess your individual risk.",
    "severity": "Severity can only be determined by a qualified healthcare professional after thorough evaluation.",
    "next_steps": "Please consult a healthcare professional for a proper diagnosis and treatment plan.",
}


def _get_confidence_level(prob):
    """Converts a raw probability to a clinical confidence descriptor."""
    if prob >= 0.85:
        return "High", colors.HexColor("#CC0000")
    elif prob >= 0.65:
        return "Moderate", colors.HexColor("#CC6600")
    elif prob >= 0.5:
        return "Low", colors.HexColor("#CC9900")
    elif prob >= 0.35:
        return "Low", colors.HexColor("#669900")
    elif prob >= 0.15:
        return "Moderate", colors.HexColor("#339933")
    else:
        return "High", colors.HexColor("#006600")

def _get_image_scaled(path, max_width):
    """Returns a ReportLab Image scaled proportionally to fit max_width, or None."""
    if not path or not os.path.exists(path):
        return None
    reader = ImageReader(path)
    w, h = reader.getSize()
    aspect = h / float(w)
    new_w = max_width
    new_h = new_w * aspect
    return Image(path, width=new_w, height=new_h)


def generate_clinical_pdf(
    patient_id,
    prediction_prob,
    uncertainty_score,
    original_img_path,
    gradcam_img_path,
    ig_img_path,
    occ_img_path,
    bb_img_path,
    hr_img_path,
    cf_img_path,
    narratives,
    similar_cases,
    output_pdf_path,
    disease_name="Pneumonia",
    include_technical=True,
    report_mode="professional",
    reference_positive_img_path=None,
    clinician_verdict=None,
    model_auc=None,
    model_accuracy=None,
    cam_img_path=None,
):
    """
    Generates a PDF Clinical Report with dual-mode support.

    Args:
        report_mode: "professional" or "public"
            - professional: Full clinical report with explainability appendix,
              validate/reject metadata, and detailed model metrics.
            - public: Simplified health screening report with disease info,
              visual comparison guide, and next-steps guidance.
        reference_positive_img_path: Path to a reference positive image
            for the visual comparison guide (public mode).
        clinician_verdict: Optional dict {"verdict": "validated"/"rejected",
            "clinician_id": str, "notes": str} for professional mode.
        model_auc: AUC-ROC score for this disease model (shown in professional mode).
        model_accuracy: Balanced accuracy for this disease model.
    """
    theme = THEME.get(report_mode, THEME["professional"])

    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()

    # ── Custom Styles (theme-aware) ──
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'],
                                  alignment=1, fontSize=18, spaceAfter=4,
                                  fontName='Helvetica-Bold')
    section_style = ParagraphStyle('SectionHead', parent=styles['Heading2'],
                                    fontSize=13, spaceAfter=6,
                                    fontName='Helvetica-Bold',
                                    textColor=colors.HexColor(theme["section_color"]))
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, leading=14, spaceAfter=6)
    body_large = ParagraphStyle('BodyLarge', parent=styles['Normal'],
                                 fontSize=12, leading=16, spaceAfter=8)
    findings_style = ParagraphStyle('Findings', parent=styles['Normal'],
                                     fontSize=10, leading=15, spaceAfter=10,
                                     fontName='Helvetica',
                                     textColor=colors.HexColor("#222222"),
                                     leftIndent=10, rightIndent=10, borderPadding=6)
    impression_style = ParagraphStyle('Impression', parent=styles['Normal'],
                                       fontSize=10, leading=14, spaceAfter=8,
                                       fontName='Helvetica-Bold',
                                       textColor=colors.HexColor(theme["section_color"]))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Italic'],
                                       textColor=colors.gray, fontSize=7.5, leading=10)
    public_disclaimer_style = ParagraphStyle('PublicDisclaimer', parent=styles['Normal'],
                                              textColor=colors.HexColor("#BF360C"),
                                              fontSize=9, leading=12,
                                              fontName='Helvetica-Bold',
                                              borderPadding=8)
    appendix_title_style = ParagraphStyle('AppendixTitle', parent=styles['Heading1'],
                                           alignment=1, fontSize=15, spaceAfter=4,
                                           fontName='Helvetica-Bold',
                                           textColor=colors.HexColor("#444444"))
    vlm_style = ParagraphStyle('VLM', parent=styles['Normal'],
                                textColor=colors.darkblue, spaceAfter=10,
                                fontName='Helvetica-Oblique', leading=14, fontSize=9.5)
    info_heading = ParagraphStyle('InfoHeading', parent=styles['Heading3'],
                                   fontSize=11, spaceAfter=4,
                                   fontName='Helvetica-Bold',
                                   textColor=colors.HexColor(theme["primary"]))

    elements = []

    # ==========================================================
    # Determine disease context
    # ==========================================================
    is_positive = prediction_prob >= 0.5
    screening_label = "POSITIVE" if is_positive else "NEGATIVE"
    confidence_level, confidence_color = _get_confidence_level(prediction_prob)

    derm_diseases = {'Melanoma', 'akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc',
                     'Actinic_Keratosis', 'Basal_Cell_Carcinoma', 'Benign_Keratosis',
                     'Dermatofibroma', 'Melanocytic_Nevus', 'Vascular_Lesion'}
    eye_diseases = {'AMD', 'Cataract', 'Diabetes', 'Diabetic_Retinopathy',
                    'Glaucoma', 'Hypertension', 'Myopia'}
    neuro_diseases = {'Dementia'}

    if disease_name in derm_diseases:
        modality_name = "Dermoscopy / Skin Lesion Imaging"
        scan_name = "Dermoscopic Image"
        anatomy_normal = "The skin lesion"
    elif disease_name in eye_diseases:
        modality_name = "Fundus Photography / Retinal Scan"
        scan_name = "Fundus Photograph"
        anatomy_normal = "The retinal structures"
    elif disease_name in neuro_diseases:
        modality_name = "Brain MRI (T1-Weighted)"
        scan_name = "Brain MRI Slice"
        anatomy_normal = "The brain parenchyma"
    else:
        modality_name = "Chest Radiograph (PA)"
        scan_name = "Chest X-Ray"
        anatomy_normal = "The lung fields"

    # ==========================================================
    # Select banner colors from theme
    # ==========================================================
    if is_positive:
        banner_bg = colors.HexColor(theme["banner_pos_bg"])
        banner_border = colors.HexColor(theme["banner_pos_border"])
        banner_text_color = theme["banner_pos_text"]
    elif prediction_prob >= 0.35:
        banner_bg = colors.HexColor(theme["banner_unc_bg"])
        banner_border = colors.HexColor(theme["banner_unc_border"])
        banner_text_color = theme["banner_unc_text"]
        screening_label = "UNCERTAIN"
    else:
        banner_bg = colors.HexColor(theme["banner_neg_bg"])
        banner_border = colors.HexColor(theme["banner_neg_border"])
        banner_text_color = theme["banner_neg_text"]

    # ==========================================================
    # PAGE 1: HEADER (both modes)
    # ==========================================================
    if report_mode == "professional":
        elements.append(Paragraph("Diagnostic Imaging Report", title_style))
    else:
        elements.append(Paragraph("Health Screening Report", title_style))

    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        f'<font size="9" color="{theme["primary"]}">{theme["mode_label"]}</font>',
        ParagraphStyle('Sub', parent=body_style, alignment=1)))
    elements.append(Spacer(1, 14))

    # ── Patient Information Table ──
    elements.append(Paragraph("Patient Information", section_style))

    info_data = [
        ["Patient ID", patient_id, "Date of Study", datetime.now().strftime('%Y-%m-%d')],
        ["Examination", modality_name, "Clinical Indication", f"{disease_name.replace('_', ' ')} Screening"],
    ]
    info_table = Table(info_data, colWidths=[110, 160, 120, 140])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#333333")),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor("#333333")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 14))

    # ── Screening Result Banner ──
    elements.append(Paragraph("Screening Result", section_style))

    if report_mode == "public":
        # Larger, friendlier result display for general public
        result_data = [[
            Paragraph(
                f'<font size="16" color="{banner_text_color}"><b>{screening_label}</b></font>'
                f'<br/>'
                f'<font size="11" color="#333333">Confidence: <b>{confidence_level}</b></font>',
                ParagraphStyle('ResultPublic', parent=body_style, alignment=1)
            )
        ]]
    else:
        result_data = [[
            Paragraph(
                f'<font size="12" color="{banner_text_color}"><b>{screening_label}</b></font>'
                f'&nbsp;&nbsp;&nbsp;'
                f'<font size="10" color="#333333">Confidence: <b>{confidence_level}</b></font>',
                body_style
            )
        ]]

    result_table = Table(result_data, colWidths=[490])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 12 if report_mode == "public" else 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12 if report_mode == "public" else 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 16))

    # ── Findings (both modes) ──
    elements.append(Paragraph("Findings", section_style))
    overall_text = narratives.get('overall', 'No findings summary available.')
    for line in overall_text.split("\n"):
        line = line.strip()
        if line:
            if report_mode == "public":
                elements.append(Paragraph(line, body_large))
            else:
                elements.append(Paragraph(line, findings_style))
    elements.append(Spacer(1, 10))

    # ── Patient Radiograph ──
    elements.append(PageBreak())
    elements.append(Paragraph("Imaging", section_style))
    elements.append(Spacer(1, 6))

    has_original = original_img_path and os.path.exists(original_img_path)

    if has_original:
        elements.append(Paragraph(f"<b>Your {scan_name}</b>", body_style))
        img_obj = _get_image_scaled(original_img_path, 380)
        if img_obj:
            elements.append(img_obj)
    elements.append(Spacer(1, 16))

    # ── Clinical Impression ──
    elements.append(Paragraph("Impression", section_style))

    clinical_impression = narratives.get('clinical_impression', '')
    if not clinical_impression:
        if is_positive:
            if prediction_prob >= 0.85:
                clinical_impression = (
                    f"Findings are highly suggestive of {disease_name.replace('_', ' ')}. "
                    f"Clinical correlation and follow-up imaging are recommended. "
                    f"Consider further workup as clinically indicated."
                )
            elif prediction_prob >= 0.65:
                clinical_impression = (
                    f"Findings are suggestive of {disease_name.replace('_', ' ')}. "
                    f"Correlation with clinical symptoms and laboratory findings is recommended."
                )
            else:
                clinical_impression = (
                    f"Findings raise the possibility of {disease_name.replace('_', ' ')}, though with limited confidence. "
                    f"Close clinical correlation is advised."
                )
        else:
            if prediction_prob <= 0.15:
                clinical_impression = (
                    f"No evidence suggestive of {disease_name.replace('_', ' ')} is identified. "
                    f"{anatomy_normal} appear unremarkable. "
                    f"Routine follow-up as clinically indicated."
                )
            elif prediction_prob <= 0.35:
                clinical_impression = (
                    f"No definitive findings of {disease_name.replace('_', ' ')} are identified. "
                    f"Clinical correlation is recommended if symptoms persist."
                )
            else:
                clinical_impression = (
                    f"Findings are equivocal for {disease_name.replace('_', ' ')}. "
                    f"Clinical correlation with patient history is strongly recommended."
                )

    elements.append(Paragraph(clinical_impression, impression_style))
    elements.append(Spacer(1, 20))

    # ── Disclaimer (both modes) ──
    if report_mode == "public":
        public_disc = (
            "⚠ IMPORTANT: This is an AI-powered health screening tool. It is NOT a medical diagnosis. "
            "Results must be confirmed by a qualified healthcare professional. Do not make medical "
            "decisions based solely on this report. If you are experiencing symptoms, please visit "
            "a doctor or hospital immediately."
        )
        elements.append(Paragraph(public_disc, public_disclaimer_style))
    else:
        clinical_disclaimer = (
            "DISCLAIMER: This report is generated by a computer-aided detection system and is intended to "
            "assist qualified medical professionals. It does NOT constitute a definitive medical diagnosis. "
            "All findings must be correlated with clinical history, physical examination, and additional "
            "diagnostic tests as deemed appropriate by the treating physician."
        )
        elements.append(Paragraph(clinical_disclaimer, disclaimer_style))

    # ==================================================================
    # PUBLIC MODE: Disease Information + Visual Comparison Guide
    # ==================================================================
    if report_mode == "public":
        elements.append(PageBreak())

        # ── Disease Information Page ──
        disease_info = DISEASE_INFO.get(disease_name, DEFAULT_DISEASE_INFO)
        elements.append(Paragraph(f"Understanding: {disease_info['name']}", section_style))
        elements.append(Spacer(1, 8))

        info_sections = [
            ("What is it?", disease_info["what"]),
            ("Common Symptoms", disease_info["symptoms"]),
            ("Risk Factors", disease_info["risk_factors"]),
            ("How serious is it?", disease_info["severity"]),
            ("Recommended Next Steps", disease_info["next_steps"]),
        ]

        for heading, content in info_sections:
            elements.append(Paragraph(heading, info_heading))
            elements.append(Paragraph(content, body_large))
            elements.append(Spacer(1, 6))

        # ── Visual Comparison Guide ──
        elements.append(PageBreak())
        elements.append(Paragraph("Visual Comparison Guide", section_style))
        elements.append(Paragraph(
            "Below is a reference image showing a confirmed positive case of this condition. "
            "Key diagnostic regions have been highlighted. You can compare this against your own "
            "image above to understand what medical professionals look for when making a diagnosis.",
            body_large))
        elements.append(Spacer(1, 12))

        if reference_positive_img_path and os.path.exists(reference_positive_img_path):
            elements.append(Paragraph(
                f"<b>Reference: Confirmed Positive Case of {disease_name.replace('_', ' ')}</b>",
                body_style))
            ref_img = _get_image_scaled(reference_positive_img_path, 400)
            if ref_img:
                elements.append(ref_img)
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(
                "<i>Highlighted regions show areas where the condition is typically visible. "
                "Note: Every case is different — the exact appearance may vary.</i>",
                ParagraphStyle('RefNote', parent=body_style,
                             textColor=colors.HexColor("#666666"), fontSize=9)))
        else:
            elements.append(Paragraph(
                "<i>Reference comparison image not available for this condition.</i>",
                body_style))

        elements.append(Spacer(1, 20))

        # ── What to tell your doctor ──
        elements.append(Paragraph("What to Tell Your Doctor", section_style))
        elements.append(Paragraph(
            f"When visiting your healthcare provider, mention that an AI screening tool "
            f"indicated a <b>{screening_label}</b> result for <b>{disease_name.replace('_', ' ')}</b> "
            f"with <b>{confidence_level}</b> confidence. Show them this report. "
            f"Your doctor may order additional tests to confirm or rule out the condition.",
            body_large))

    # ==================================================================
    # PROFESSIONAL MODE: Technical Explainability Appendix
    # ==================================================================
    if report_mode == "professional" and include_technical:
        elements.append(PageBreak())

        elements.append(Paragraph("Technical Appendix", appendix_title_style))
        elements.append(Paragraph(
            '<font size="9" color="#888888">AI Model Explainability &amp; Interpretability Details</font>',
            ParagraphStyle('AppSub', parent=body_style, alignment=1)))
        elements.append(Spacer(1, 16))

        # ── Model Information ──
        elements.append(Paragraph("Model Information", section_style))

        model_data = [
            ["Target Condition", disease_name.replace('_', ' ')],
            ["Raw Probability", f"{prediction_prob:.4f} ({prediction_prob:.2%})"],
            ["Screening Decision", f"{'POSITIVE' if is_positive else 'NEGATIVE'} (threshold = 0.50)"],
            ["Uncertainty (MC Dropout ±σ)", f"{uncertainty_score:.4f} ({uncertainty_score:.2%})"],
        ]
        if model_auc is not None:
            model_data.append(["Model AUC-ROC (Test Set)", f"{model_auc:.4f}"])
        if model_accuracy is not None:
            model_data.append(["Balanced Accuracy", f"{model_accuracy:.2f}%"])
        model_data.append(["Architecture", "SwinV2-S + MultiModalFusion (50M params)"])
        model_data.append(["Training", "Balanced 1:1 undersampling, Focal Loss (γ=2.0)"])
        model_data.append(["Inference Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        model_table = Table(model_data, colWidths=[180, 340])
        model_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#333333")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(theme["table_header_bg"])),
        ]))
        elements.append(model_table)
        elements.append(Spacer(1, 16))

        # ── Clinician Verdict (if provided) ──
        if clinician_verdict:
            elements.append(Paragraph("Clinician Feedback", section_style))
            verdict = clinician_verdict.get("verdict", "pending").upper()
            verdict_color = "#006600" if verdict == "VALIDATED" else "#CC0000"
            verdict_data = [
                ["Clinician Verdict", Paragraph(
                    f'<font color="{verdict_color}"><b>{verdict}</b></font>', body_style)],
                ["Clinician ID", clinician_verdict.get("clinician_id", "Anonymous")],
                ["Notes", clinician_verdict.get("notes", "—")],
                ["Timestamp", clinician_verdict.get("timestamp",
                                                      datetime.now().strftime('%Y-%m-%d %H:%M:%S'))],
            ]
            verdict_table = Table(verdict_data, colWidths=[120, 400])
            verdict_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ]))
            elements.append(verdict_table)
            elements.append(Spacer(1, 16))

        # ── Annotated Visualizations Grid (2x2) ──
        elements.append(Paragraph("Annotated Visualizations", section_style))

        has_annotated = bb_img_path and os.path.exists(bb_img_path)
        has_hr = hr_img_path and os.path.exists(hr_img_path)
        has_cf = cf_img_path and os.path.exists(cf_img_path)

        if has_original and has_annotated and has_hr and has_cf:
            grid_data = [
                [f"Original Patient {scan_name}", "Global Region Output (SAS)"],
                [Image(original_img_path, width=200, height=200),
                 Image(bb_img_path, width=200, height=200)],
                ["High-Res Exact Edges (IG)", "Generative Counterfactual"],
                [Image(hr_img_path, width=200, height=200),
                 Image(cf_img_path, width=200, height=200)]
            ]
            grid_widths = [220, 220]
        elif has_original and has_annotated:
            grid_data = [
                [f"Original Patient {scan_name}", "Annotated AI Findings"],
                [Image(original_img_path, width=220, height=220),
                 Image(bb_img_path, width=220, height=220)]
            ]
            grid_widths = [240, 240]
        elif has_original:
            grid_data = [
                [f"Original Patient {scan_name}", "Annotated AI Findings"],
                [Image(original_img_path, width=220, height=220), "Not Available"]
            ]
            grid_widths = [240, 240]
        else:
            grid_data = [["Not Available", "Not Available"]]
            grid_widths = [240, 240]

        grid_table = Table(grid_data, colWidths=grid_widths)
        grid_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(grid_table)
        elements.append(Spacer(1, 14))

        # ── Interpretability Guide ──
        elements.append(Paragraph("Interpretability Guide", styles['Heading3']))
        guide_text = (
            "<b>Global Region Output (SAS):</b> Uses a broad yellow distinct outline to highlight the massive macro-structures "
            "and general regions of concern the network's attention layers focused on.<br/><br/>"
            "<b>High-Res Exact Edges (IG):</b> This traces magenta polygons precisely hugging the exact "
            "bone edges, lung opacities, or anatomical artifacts down to the pixel level. "
            "This details exactly what distinct anatomical fragments trigger the model.<br/><br/>"
        )
        elements.append(Paragraph(guide_text, body_style))
        elements.append(Spacer(1, 10))

        # ── RAG Historical Cases ──
        if similar_cases and len(similar_cases) > 0:
            elements.append(Paragraph("Retrieval-Augmented Historical Context (RAG)", section_style))
            rag_desc = "The AI referenced its training database to find the top 3 most mathematically similar historical cases."
            elements.append(Paragraph(rag_desc, body_style))
            elements.append(Spacer(1, 8))

            rag_data = [["Match Rank", "Cosine Similarity", "Historical Ground Truth"]]
            for i, case in enumerate(similar_cases):
                sim_pct = f"{case['similarity_score']:.1%}"
                gt_truth = case['diagnosis'].upper()
                rag_data.append([f"#{i+1}", sim_pct, gt_truth])

            rag_table = Table(rag_data, colWidths=[100, 150, 200])
            rag_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(rag_table)
            elements.append(Spacer(1, 16))

        # ── AI Transparency: Where the System Looked ──
        elements.append(PageBreak())
        elements.append(Paragraph("AI Transparency: Where the System Looked", section_style))
        elements.append(Paragraph(
            "The coloured overlays below show which parts of your scan the AI system focused on when making its decision. "
            "For the first three maps, darker blue regions highlight areas the system considered most important, while lighter areas "
            "had less influence. The fourth map (Activation Heatmap) uses a different colour scale where red and warm colours "
            "indicate stronger focus. These maps help verify that the AI is looking at the right part of the image, "
            "rather than making a decision based on irrelevant areas.", body_style))
        elements.append(Spacer(1, 10))

        def add_heatmap_section(title, desc, path, narrative_key, do_break=True, max_width=450):
            """Adds a full-page heatmap analysis section to the appendix."""
            if do_break:
                elements.append(PageBreak())

            elements.append(Paragraph(title, styles['Heading3']))
            elements.append(Paragraph(desc, body_style))
            elements.append(Spacer(1, 8))

            img_obj = _get_image_scaled(path, max_width)
            if img_obj:
                elements.append(img_obj)
            else:
                elements.append(Paragraph("<i>Image file missing.</i>", body_style))
            elements.append(Spacer(1, 16))

            # VLM narrative for this specific heatmap
            elements.append(Paragraph("AI Reasoning Analysis:", styles['Heading4']))
            spec_text = narratives.get(narrative_key, 'No specific analysis generated.')
            for line in spec_text.split("\n"):
                line = line.strip()
                if line:
                    elements.append(Paragraph(line, vlm_style))
            elements.append(Spacer(1, 16))

        add_heatmap_section(
            "1. Attention Focus Map",
            "This overlay shows which areas of your scan the AI paid the most attention to. "
            "The system processes the image in small patches and learns which patches are related to each other. "
            "Darker blue regions are where the system found the strongest visual evidence relevant to the diagnosis.",
            gradcam_img_path,
            "gradcam",
            do_break=False
        )

        add_heatmap_section(
            "2. Pixel Importance Map",
            "This map highlights the individual pixels and fine details that contributed most to the AI's decision. "
            "It reveals specific edges, textures, or patterns the system considered significant — providing a more "
            "detailed view than the broader attention map above.",
            ig_img_path,
            "ig"
        )

        add_heatmap_section(
            "3. Region Sensitivity Test",
            "This test was performed by covering different parts of the scan one at a time and measuring how much "
            "the AI's confidence changed. Darker blue patches are areas the system relies on most \u2014 when these were hidden, "
            "the AI became much less certain about its diagnosis. This independently confirms which regions matter most.",
            occ_img_path,
            "occlusion"
        )

        if cam_img_path and os.path.exists(cam_img_path):
            add_heatmap_section(
                "4. Activation Heatmap",
                "This heatmap directly shows which areas of the scan triggered the strongest response inside the AI system. "
                "Unlike the other maps which analyse the decision after it is made, this one captures what the system was "
                "\"seeing\" during the actual diagnosis — similar to highlighting parts of a document that caught a reader's eye. "
                "This same method is used on the mobile app for instant on-device transparency.",
                cam_img_path,
                "gradcam",
                max_width=350
            )

        # ── Technical Disclaimer ──
        elements.append(Spacer(1, 24))
        tech_disclaimer = (
            "TECHNICAL DISCLAIMER: This appendix contains internal machine learning diagnostics generated by "
            "an automated Swin Transformer V2 model. These visualizations and metrics are intended for research "
            "and explainability analysis only. They are NOT a substitute for professional medical diagnosis or "
            "clinical evaluation."
        )
        elements.append(Paragraph(tech_disclaimer, disclaimer_style))

    # Build Document
    doc.build(elements)
    return output_pdf_path
