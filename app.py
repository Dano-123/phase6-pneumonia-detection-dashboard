# -*- coding: utf-8 -*-"
"""
# -----------------------------------------------------------------------------
Author: Daniel O’Keeffe
Module: Phase 6 Inference Module
Date: September 23rd 2026
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
Purpose of Script:
This application implements the Phase 6 clinical-facing dashboard for the
pneumonia‑detection system. It provides an interactive Streamlit interface
that enables clinicians to upload chest X‑ray images and receive model-driven
diagnostic support in real time.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
The dashboard integrates model inference, explainability visualisation,
clinical interpretation, and structured radiology-style reporting.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
Core Functional Components
# -----------------------------------------------------------------------------

1. Model Loading
   - Loads the trained Phase 5 CNN.
   - Prepares the model for inference within the Streamlit runtime.

2. Image Upload & Prediction
   - Accepts user-uploaded chest X‑ray images.
   - Generates predicted class NORMAL or PNEUMONIA.
   - Computes class probabilities and overall model confidence.

3. Explainability Visualisation
   - Produces Score‑CAM and SmoothGrad heatmaps.
   - Generates a blended CAM (Score‑CAM + SmoothGrad).
   - Applies a placeholder lung segmentation mask for region-of-interest focus.
   - Creates an overlay image combining CAM with the original X‑ray.

4. Clinical Interpretation
   - Generates structured clinical notes based on model output.
   - Classifies heatmap activation pattern such as unilateral-left, unilateral-right,
     bilateral, and diffuse.
   - Estimates severity such as Mild, Moderate, Severe.
   - Produces radiology-style findings and impression text.

5. Dashboard Presentation
   - Displays probability bar charts, confidence indicators, severity markers,
     model metadata, and interpretability visuals.
   - Organises results into an imaging panel and clinical summary panel.
   - Provides downloadable PDF radiology reports with embedded images,
     findings, impression, and patient-specific identifiers.

"""
# ----------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_inference import load_model
from model_inference import predict_image
from model_inference import generate_blended_cam
from model_inference import overlay_cam
from model_inference import classify_heatmap_pattern
from model_inference import classify_severity

from fpdf import FPDF 
import tempfile 
import datetime
import random

# ----------------------------------------------------------------------
# PDF Report Generator # added on 25/09/26
# ----------------------------------------------------------------------
class RadiologyPDF(FPDF):
    def header(self):
        # Set main report title in bold, centered
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Radiology Report - Pneumonia Detection", ln=True, align="C")
        self.ln(3)
        
        # Draw horizontal divider line under header
        self.set_draw_color(150, 150, 150)
        self.line(10, 28, 200, 28)
        
        # Add spacing before body content
        self.ln(6)
        
    def footer(self):
        # Position footer 15px from bottom
        self.set_y(-15)
        
        # Set font
        self.set_font("Arial", "", 10)
        
        # Light grey footer text
        self.set_text_color(120, 120, 120)
        
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        # Section heading formatting
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True)
        
        # Draw thin divider line under section title
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        
        # Add spacing before section content
        self.ln(6)


def create_pdf_report(label, normal_prob, pneu_prob, confidence,
                      heatmap_pattern, severity, findings, impression, notes,
                      image, cam_image, overlay_image, bar_chart_image, 
                      patient_id):
    """
    Generate a radiology PDF report for pneumonia detection, combining model outputs,
    clinical narrative, and imaging visualisations.
    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    - label : str
        Predicted class label: NORMAL or PNEUMONIA.
    - normal_prob : float
        Model probability for the NORMAL class.
    - pneu_prob : float
        Model probability for the PNEUMONIA class.
    - confidence : float
        Overall model confidence score for the prediction.
   -  heatmap_pattern : str
        Text description of the CAM/Grad-CAM heatmap pattern.
    - severity : str
        Clinical severity level: Mild, Moderate, Severe.
    - findings : str
        Detailed radiology findings narrative.
    - impression : str
        Final diagnostic impression or summary statement.
    - notes : list[str]
        List of interpretation notes or bullet points to include in the report.
    - image : PIL.Image.Image
        Original chest X-ray image.
    - cam_image : PIL.Image.Image
        CAM/Grad-CAM heatmap image.
    - overlay_image : PIL.Image.Image
        Overlay image combining X-ray and CAM heatmap.
    - bar_chart_image : PIL.Image.Image
        Bar chart image showing model probabilities or severity distribution.
    - patient_id : str
        Unique patient identifier to be displayed in the report.
    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------
    bytes
        The generated PDF report as a bytes object, suitable for download or storage.
    """

    pdf = RadiologyPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

# ----------------------------------------------------------------------
# Clinical Summary
# ----------------------------------------------------------------------
    pdf.section_title("Clinical Summary")
    pdf.set_font("Arial", "", 12)
    
    # Patient ID
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient ID:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, patient_id)
    pdf.ln(4)
    
    # Prediction Result
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Prediction Result:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, label)
    pdf.ln(4)
    
    # Normal Probability
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Normal Probability:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, "%.3f" % normal_prob)
    pdf.ln(4)
    
    # Pneumonia Probability
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Pneumonia Probability:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, "%.3f" % pneu_prob)
    pdf.ln(4)

    # Model Confidence
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Model Confidence:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, "%.3f" % confidence)
    pdf.ln(4)

    # Heatmap Pattern
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Heatmap Pattern:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, heatmap_pattern)
    pdf.ln(4)

    # Severity Level
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Severity Level:")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, severity)
    pdf.ln(4)
    
 # ----------------------------------------------------------------------
 # Create PDF Object
 # ----------------------------------------------------------------------
    pdf.add_page()                                     # initialise PDF object and first page
    pdf.set_auto_page_break(auto=True, margin=15)      # enable automatic page breaks
    pdf.section_title("Imaging")                      
    
# ----------------------------------------------------------------------
# Creating Image for PDF 
# ----------------------------------------------------------------------
    # Save PIL image to a temporary PNG file because FPDF requires a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp1:
        image.save(tmp1.name)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Original Chest X-ray", ln=True)  # Section label
        pdf.image(tmp1.name, w=120)                      # embed chest x-ray image
        pdf.ln(6)                                        # spacing after image

    # Save CAM heatmap to temporary file for embedding
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp2:
        cam_image.save(tmp2.name)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "CAM Heatmap", ln=True)
        pdf.image(tmp2.name, w=120)
        pdf.ln(6)

    # Save overlay image showing model attention regions
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp3:
        overlay_image.save(tmp3.name)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Overlay (X-ray + CAM)", ln=True)
        pdf.image(tmp3.name, w=120)
        pdf.ln(6) 
    
    # Section header for the probability visualisation page
    pdf.section_title("Probability Bar Chart")
    
    # Save bar chart (PIL image) to a temporary PNG file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp4:
        bar_chart_image.save(tmp4.name)
        pdf.image(tmp4.name, w=120)
    pdf.ln(6)
    
# ----------------------------------------------------------------------
# Chest X-ray analysis information for PDF document
# ----------------------------------------------------------------------

    # Findings
    pdf.add_page()                            # new page for narrative sections
    pdf.section_title("Findings")             # section heading with divider
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, findings)            # findings text
    pdf.ln(6)                                 # spacing before next section

    # Impression
    pdf.section_title("Impression")           # diagnostic impression heading
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, impression)          # final diagnostic summary
    pdf.ln(6)
  
    # Interpretation Notes
    pdf.section_title("Interpretation Notes")  # bullet-point interpretation notes
    pdf.set_font("Arial", "", 12)
    
    # Loop through each note and render as a bullet point
    for n in notes:
        pdf.multi_cell(0, 8, "- " + n)
        pdf.ln(2)

    return pdf.output(dest="S").encode("latin-1")

# ----------------------------------------------------------------------
# Dynamic clinical interpretation module
# ----------------------------------------------------------------------
def generate_clinical_interpretation(
    pred_class: str,
    pneumonia_prob: float,
    normal_prob: float,
    confidence: float,
    heatmap_pattern: str = "unknown",
    severity: str = "unknown",          
) -> list[str]:
    """
    This function enhances interpretation by providing clinically relevant 
    explanations.
    Generates structured clinical interpretation notes based on:
    - Predicted class (NORMAL or PNEUMONIA)
    - Class probabilities
    - Model confidence
    - Heatmap activation pattern (unilateral / bilateral / diffuse)
    - Severity of activation (Mild / Moderate / Severe)  # added on 22/09/2026
    """

    notes = []
    
# ----------------------------------------------------------------------
# Diagnosis-specific messaging # added on 22/09/26
# ----------------------------------------------------------------------
    if pred_class.upper() == "PNEUMONIA":
        notes.append("The prediction system detected radiographic patterns consistent with pneumonia."
                     )
        
# ----------------------------------------------------------------------
# Heatmap interpretation on left & right lungs # added on 23/09/26
# ----------------------------------------------------------------------
        if heatmap_pattern == "Unilateral_left":
            notes.append("Heatmap highlights unilateral consolidation in the left lung field.")
        elif heatmap_pattern == "Unilateral_right":
            notes.append("Heatmap highlights unilateral consolidation in the right lung field.")
        elif heatmap_pattern == "Bilateral":
            notes.append("Heatmap highlights bilateral inflammatory involvement across both lung fields.")
        else:
            notes.append("Heatmap highlights localized regions of consolidation within the lung parenchyma.")

        # Severity messaging
        notes.append(f"Estimated severity of activation: {severity}.")
    else:
        # NORMAL case interpretation
        notes.append("No strong radiographic evidence of pneumonia detected by the model.")
        notes.append("Heatmap shows low-intensity activation across lung fields, consistent with a normal pattern.")
        
# ----------------------------------------------------------------------
# Confidence-based messaging
# ----------------------------------------------------------------------    
    if confidence < 0.70:
        notes.append("Confidence is limited. Recommend cautious interpretation and close clinical correlation.")
    elif confidence >= 0.90:
        notes.append("High confidence supports the current radiographic assessment. Clinical correlation remains essential")
    else:
        notes.append("Moderate confidence. Findings should be interpreted alongside clinical history and examination.")
        
# ----------------------------------------------------------------------
# Safety disclaimer
# ----------------------------------------------------------------------
    notes.append("Model generated output is supportive and does not replace clinical judgement or formal radiology reporting")
    
    return notes

# ----------------------------------------------------------------------
# Radiology-style findings and impression generator
# ----------------------------------------------------------------------
def generate_findings_and_impression(pred_class, heatmap_pattern, severity, confidence):
    
    """
    Generate structured radiology-style 'Findings' and 'Impression' text
    based on CAM activation patterns, severity, and model confidence.

    Parameters:
        pred_class (str):
            Model-predicted class ("NORMAL" or "PNEUMONIA").
        heatmap_pattern (str):
            CAM activation pattern classification:
            - "Bilateral"
            - "Unilateral left"
            - "Unilateral right"
            - "Diffuse" or other fallback category.
        severity (str):
            Estimated severity of activation ("Mild", "Moderate", "Severe").
        confidence (float):
            Model confidence score between 0 and 1.

    Returns:
        tuple[str, str]:
            findings (str): Combined radiology-style findings text.
            impression (str): High-level clinical impression statement.
    """
    
    findings_lines = []
    
# ----------------------------------------------------------------------
# Describe CAM activation pattern across lung fields
# ----------------------------------------------------------------------
    if heatmap_pattern == "Bilateral":
        findings_lines.append("Bilateral CAM activation in mid-lower lung zones.")
    elif heatmap_pattern == "Unilateral_left":
        findings_lines.append("Unilateral CAM activation predominantly in the left lung field.")
    elif heatmap_pattern == "Unilateral_right":
        findings_lines.append("Unilateral CAM activation predominantly in the right lung field.")
    else:
        findings_lines.append("Diffuse CAM activation without clear focal consolidation.")
        
# ----------------------------------------------------------------------
# Add severity and confidence details
# ----------------------------------------------------------------------
    findings_lines.append(f"Estimated severity: {severity}.")
    findings_lines.append(f"Model confidence: {confidence:.3f}.")
    
    # Combine list into a single radiology-style findings paragraph
    findings = " ".join(findings_lines)
    
# ----------------------------------------------------------------------
# Impression section: Diagnosis-level summary
# ----------------------------------------------------------------------  
    if pred_class.upper() == "PNEUMONIA":
        impression = ("Model generated findings are consistent with pneumonia. Recommend correlation with clinical symptoms.")
    else:
        impression = ("No radiographic evidence of pneumonia identified."
                     "Recommend routine clinical correlation.")
        
        
    return findings, impression
# ----------------------------------------------------------------------
# Lung segmentation placeholder
# ----------------------------------------------------------------------
def generate_lung_mask(cam):
    
    """
    Generate a simple rectangular lung mask for the CAM heatmap.

    This placeholder segmentation approximates the lung fields using fixed
    proportional boundaries. It is not anatomically precise, but provides
    a controlled region-of-interest for highlighting CAM activation.

    Parameters:
        cam (np.ndarray):
            2D CAM heatmap array (H x W).

    Returns:
        np.ndarray (bool):
            Boolean mask of the same shape as `cam`, where True indicates
            pixels inside the approximated lung region.
    """
    # Extract height and width of the CAM heatmap
    h, w = cam.shape
    
    # Initialise empty boolean mask
    mask = np.zeros_like(cam, dtype=bool)
    
    # Define approximate lung boundaries using proportional offsets
    left = int(w * 0.2)
    right = int(w * 0.8)
    top = int(h * 0.15)
    bottom = int(h * 0.85)
    
    # Fill mask region corresponding to the lung area
    mask[top:bottom, left:right] = True
    
    return mask

def apply_lung_mask(cam):
    """
    Apply the lung mask to the CAM heatmap.

    Parameters:
        cam (np.ndarray):
            2D CAM heatmap array (H x W).

    Returns:
        np.ndarray:
            CAM heatmap with all values outside the lung mask set to zero.
    """
    # Generate lung mask using placeholder segmenatation
    mask = generate_lung_mask(cam)
    
    # Create masked CAM output
    cam_masked = np.zeros_like(cam)
    cam_masked[mask] = cam[mask]
    
    return cam_masked

# ----------------------------------------------------------------------
# Streamlit App layout
# ----------------------------------------------------------------------
st.set_page_config(page_title="Pneumonia Detection Dashboard", layout="wide")

# ----------------------------------------------------------------------
# Dashboard header
# ----------------------------------------------------------------------
st.markdown("""
# 🩺 Clinical Pneumonia Detection Dashboard
Upload a chest X ray to generate prediction, probability, confidence, and interpretability.
""")

# ----------------------------------------------------------------------
# Displays static information about deployed CNN model used in Phase 6
# ----------------------------------------------------------------------
with st.expander("📊 Model Metadata"):
    st.markdown("""
    **Model:** Phase 5 CNN
    **Explainability:** Score-CAM & SmoothGrad (blended)
    **Image Size:** 128 X 128
    **Deployment Phase:** Phase 6
    """)
    
# ----------------------------------------------------------------------
# Load trained CNN model from Phase 5
# ----------------------------------------------------------------------
model = load_model()

# ----------------------------------------------------------------------
# File upload of X-ray images
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload chest X-ray image", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------------------------
# Inference
# - Executes the full prediction pipeline once X-ray image is uploaded.
# - This includes: Model inference, CAM generation, lung masking,
#   explainability visualisation, probability charts, confidence gauge,
#   severity classifcation, and clinical interpretation.
# ----------------------------------------------------------------------
if uploaded_file is not None:
    
    # Load uploaded chest X-ray image using PIL
    image = Image.open(uploaded_file)
    
    # Create two-column layout:left = imaging, right = clinical summary
    left_col, right_col = st.columns([1.3, 1])
    
# ----------------------------------------------------------------------
# Left column: Imaging panel
# ----------------------------------------------------------------------   
    with left_col:
        st.markdown("### 🖼 Imaging Panel")
         
        # Display original chest X-ray image                                    
        st.image(image, caption="Original chest X-ray")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Run model inference: Returns class label, probabilities, confidence, and tensor
        label, pneu_prob, normal_prob, confidence, img_tensor = predict_image(model, image)
        
        # Generate blended Score-CAM & SmoothGrad heatmap
        cam_raw = generate_blended_cam(model, img_tensor)
        
        # Apply placeholder lung segmentation mask to CAM
        cam = apply_lung_mask(cam_raw)
        
        # Display masked CAM heatmap
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size) #added 24/9/26
        #st.image(cam, caption="Blended CAM Heatmap")
        st.image( cam_resized, caption="Blended CAM Heatmap") #added 24/9/26
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Overlay CAM heatmap on original chest X-ray image
        overlay_img = overlay_cam(image, cam)
        st.image(overlay_img, caption="Overlay (Chest X-ray & CAM)") 
        
# ----------------------------------------------------------------------
# Right Column: Clinical Summary 
# ----------------------------------------------------------------------  
    with right_col:
        st.markdown("### 📋 Clinical Summary")
        
        # Probability Distribution
        st.markdown("#### Probability Distribution")
        
        # Create bar chart figure #ADDED ON 25/09/26
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.bar(["Normal", "Pneumonia"], [normal_prob, pneu_prob], color=["#4CAF50", "#F44336"])
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle = "--", alpha=0.6)
        st.pyplot(fig)
        
        # Save chart to a temporary PNG file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
            fig.savefig(tmp_chart.name, format="png", dpi=300)
            bar_chart_image = Image.open(tmp_chart.name)
        
# ----------------------------------------------------------------------
# Bar chart showing NORMAL vs PNEUMONIA probabilities
# ----------------------------------------------------------------------
        # Confidence Gauge
        st.markdown("#### Confidence Gauge")   
        
# ----------------------------------------------------------------------
# Displays confidence level using colour-coded indicators
# ----------------------------------------------------------------------
        if confidence < 0.70:
            st.markdown(f"🟠 Low confidence ({confidence:.3f})")
        elif confidence >= 0.90:
            st.markdown(f"🟢 High confidence ({confidence:.3f})")
        else:
            st.markdown(f"🟡 Moderate confidence ({confidence:.3f})")  
            
# ----------------------------------------------------------------------
# Heatmap Pattern & Severity
# ----------------------------------------------------------------------
        heatmap_pattern = classify_heatmap_pattern(cam)
        #severity = classify_severity(cam)  
        severity = classify_severity(cam, label) # added 23/09/26
        
        st.markdown("#### Severity Level")
        if severity == "Mild":
            st.markdown("🟩 Mild")
        elif severity == "Moderate":
            st.markdown("🟨 Moderate")
        else:
            st.markdown("🟥 Severe")
            
# ----------------------------------------------------------------------
# Prediction Result
# ----------------------------------------------------------------------
        result_color = "🟢 NORMAL" if label == "NORMAL" else "🔴  PNEUMONIA"
        st.markdown(f"### Prediction Result: {result_color}")

        # Interpretation Notes
        st.markdown("### 📝 Interpretation Notes")
        
        # Generate structured clinical interpretation notes
        notes = generate_clinical_interpretation(
            label, pneu_prob, normal_prob, confidence, heatmap_pattern, severity
        )
        
        # Display each interpretation bullet point
        for n in notes:
            st.write(f"• {n}")    
            
# ----------------------------------------------------------------------
# Radiology-style Report
# ----------------------------------------------------------------------
        # Section heading for the radiology-style output
        st.markdown("### 🩺 Radiology - Report")
        
        # Generate Findings & Impression text
        findings, impression = generate_findings_and_impression(
            label, heatmap_pattern, severity, confidence
        )
        st.markdown("**Findings:**")                 # bold section label
        st.write(findings)                           # display findings
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        
        st.markdown("**Impression:**")
        st.write(impression) 
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Unique Patient ID that combines date, time, and a random 4-digit number for uniqueness
        patient_id = f"PNEU-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}-{random.randint(1000,9999)}"
            
# ----------------------------------------------------------------------
# PDF Download Button
# ----------------------------------------------------------------------
# - Generate PDF bytes using the full radiology report builder.
# - Passes all clinical metadata, model outputs, and imaging assets.
# - Streamlit download button for exporting the radiology report.
# ----------------------------------------------------------------------

        pdf_bytes = create_pdf_report(
            label, normal_prob, pneu_prob, confidence,
            heatmap_pattern, severity, findings, impression, notes,
            image, cam_resized, overlay_img, bar_chart_image, patient_id)
        
        st.download_button(
            label="📥 Download Radiology Report (PDF)",   # button label with icon
            data=pdf_bytes,                               #  PDF bytes returned from generator
            file_name=f"{patient_id}.pdf",                # patient specific filename
            mime="application/pdf"
        ) 
        
# Footer
st.markdown("---")                                                            # visual divider
st.markdown("**G13 Healthcare Analytics Project 6 — Phase 6 Deployment**")