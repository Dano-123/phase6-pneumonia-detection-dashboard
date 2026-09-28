G13 Healthcare Analytics Project 6: Phase 6 Deployment 
Clinical Pneumonia Detection Dashboard (Phase 6)
This repository contains the Phase 6 cloud deployment version of the pneumonia detection dashboard. It provides a clinician facing interface for uploading chest X ray images and generating:
•	Model predictions (NORMAL vs PNEUMONIA)
•	Probability distribution
•	Confidence score
•	Explainability visualisations (Score CAM + SmoothGrad)
•	Lung masked CAM heatmaps
•	Overlay images
•	Severity classification
•	Structured clinical interpretation
•	Downloadable radiology style PDF reports
This version is optimised for Streamlit Cloud and contains only the files required for cloud execution.
Repository Structure (Cloud Deployment)
APP/
│
├── app.py                     # Streamlit Phase 6 dashboard
├── model_inference.py         # Phase 5 CNN + CAM + explainability pipeline
├── P5_CNN_Model_WEIGHTS.pth   # Trained model weights (CPU-compatible)
├── requirements.txt           # Cloud-safe dependency list
└── README.md                  # This documentation file
Only essential deployment files are included. 

 Phase 6 Deployment Overview
1. Model Inference (Phase 5 CNN)
The dashboard loads the trained Phase 5 convolutional neural network and performs:
•	Preprocessing (grayscale → resize → normalise)
•	Forward inference
•	Softmax probability computation
•	Confidence scoring
•	Tensor extraction for CAM generation

2. Explainability (Score CAM + SmoothGrad)
The system generates:
•	Score CAM activation maps
•	SmoothGrad saliency maps
•	Blended CAM (70% Score CAM, 30% SmoothGrad)
•	Lung masked CAM using a proportional segmentation mask
•	Overlay images combining CAM + original X ray

3. Clinical Interpretation Module
The dashboard produces:
•	Heatmap pattern classification
o	Bilateral
o	Unilateral_left
o	Unilateral_right
o	Diffuse
•	Severity estimation
o	Mild
o	Moderate
o	Severe
•	Structured clinical notes
•	Radiology style findings
•	Diagnostic impression

4. Radiology PDF Report Generator
The system generates a downloadable PDF containing:
•	Patient ID
•	Prediction result
•	Probabilities
•	Confidence
•	Heatmap pattern
•	Severity
•	Original X ray
•	CAM heatmap
•	Overlay image
•	Probability bar chart
•	Findings
•	Impression
•	Interpretation notes
Temporary files are handled using:
•	tempfile.NamedTemporaryFile(dir=".", delete=False)
•	This ensures full compatibility with Streamlit Cloud

5. Cloud Deployment Instructions
1. Push the APP folder to GitHub
Ensure the folder structure is exactly:
APP/
   app.py
   model_inference.py
   P5_CNN_Model_WEIGHTS.pth
   requirements.txt
   README.md
2. Deploy on Streamlit Cloud
1.	Go to: https://streamlit.io/cloud
2.	Select Deploy App
3.	Choose your GitHub repository
4.	Set the entry point to:
APP/app.py
5.	Deploy
6.	Streamlit Cloud will automatically install dependencies from requirements.txt

requirements.txt 
Cloud deployment uses lightweight, CPU compatible libraries:
streamlit==1.36.0
torch==2.2.0
torchvision==0.17.0
numpy==1.26.4
matplotlib==3.8.2
Pillow==10.2.0
fpdf==1.7.2

6.Cloud testing
Upload any chest X ray image (JPG/JPEG/PNG). Verify:
•	Prediction
•	CAM heatmap
•	Overlay
•	Severity
•	Clinical notes
•	PDF download
Everything should run end to end.
