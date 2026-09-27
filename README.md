APP: Phase 6 Web Application Development & Model Deployment
This folder contains the complete Phase 6 deployment module, implementing a clinical facing web application for pneumonia detection from chest X ray images. The system is built using Streamlit, integrates the finalized prediction model from Phase 5, and provides real time inference, visualisation, and structured clinical interpretation.

Folder Structure
Code
APP/
│
├── app.py                       # Streamlit web application
├── model_inference.py           # Inference pipeline (preprocessing, prediction, CAM)
├── P5_CNN_Model_WEIGHTS.pth     # Trained pneumonia classifier (Phase 5)
└── requirements.txt             # Python dependencies

Phase 6 Objectives

1. Develop a user-friendly web application using Streamlit
•       The app.py file provides a clean, intuitive interface suitable for clinicians and examiners. It supports drag and drop image upload, real time inference, and visualisation.

2. Allow users to upload a chest X-ray image
•       The dashboard includes an upload widget that accepts .png and .jpg chest X ray images.

3. Automatically preprocess the uploaded image and apply the finalized prediction model
model_inference.py handles:
•	Image resizing and normalisation
•	Tensor conversion
•	Loading the Phase 5 CNN model
•	Running forward inference
•	Generating Grad CAM activation maps
•	Applying lung segmentation masks (if enabled)

4. Display the predicted class (Normal/Pneumonia) with probability and confidence
The UI presents:
•	Predicted class
•	Probability distribution
•	Confidence level
•	Severity estimate (mild / moderate / severe)

5. Provide visualisation of the uploaded X-ray and summary of prediction results
The dashboard displays:
•	Original X ray
•	Lung segmentation mask
•	Grad CAM heatmap
•	Blended activation overlay
•	Structured clinical interpretation (Findings + Impression)

6. Deploy the application for local or cloud-based access and validate end-to-end performance
The app runs locally via Streamlit and is structured for easy deployment to:
•	Streamlit Cloud
•	Azure App Service
•	Docker containers
•	Any Python based hosting environment
End to end validation was performed to ensure correct behaviour from image upload to preprocessing to inference to visualisation to interpretation.

3. Running the Application
•       Before running the application, navigate to the APP directory using cd:
•       cd "C:\Users\admin\Desktop\G13_Healthcare_Analytics_Project_6_Final_Submission\APP"

Install Dependencies
•       python -m pip install -r requirements.txt
Start the dashboard :
•       python -m streamlit run app.py
The interface will open automatically in your browser.

4. Inference Pipeline Overview
The inference engine (model_inference.py) performs:
•	Preprocessing
o	Normalise
o	Convert to tensor
•	Model Loading
o	Loads the Phase 5 CNN classifier (P5_CNN_Model_WEIGHTS.pth)
•	Prediction
o	Computes Normal vs Pneumonia probabilities
o	Returns predicted class + confidence
•	Grad CAM Activation Mapping
o	Highlights pathological regions
o	Optional lung mask applied for clinical clarity
•	Severity Classification
o	Mild / Moderate / Severe
o	Based on masked activation intensity
•	Clinical Interpretation
o	Radiology style structured report
o	Findings + Impression + Confidence

5. Streamlit UI Features
The dashboard provides:
•	Image upload panel
•	Display of:
o	Original X ray
o	Lung mask
o	Grad CAM heatmap
o	Blended activation overlay
•	Probability and confidence
•	Severity indicator
•	Clinical grade interpretation summary
•	Footer with project details

6. Safety Notice
This system provides computer assisted interpretation only. It does not replace radiological expertise or clinical judgement.

7. Option A: Streamlit Cloud
•	Upload the APP folder to a GitHub repository
•	Go to share.streamlit.io
•	Select your repository
•	Set the entry point to:
        APP/app.py
•	Deploy
•	A public URL is generated automatically

Option B: Azure App Service
•	Package the app using Docker
•	Deploy container to Azure
•	Configure port 8501

Option C: Docker Container
•       Create a Dockerfile and run:
•       docker build -t pneumonia-app .
•       docker run -p 8501:8501 pneumonia-app

Option D: Any Python Hosting Environment
• Heroku
• Railway
• Google Cloud Run