# -*- coding: utf-8 -*-
"""
# -----------------------------------------------------------------------------
Author: Daniel O’Keeffe
Module: Phase 6 Inference Module
Date: September 23rd 2026
Purpose of Script:
# -----------------------------------------------------------------------------
This script performs full inference for the Phase 6 deployment pipeline of the
pneumonia‑detection system. It loads the trained Phase 5 CNN model and applies
multiple explainability methods to single chest X-ray images, including:
    - Grad-CAM (baseline).
    - Score-CAM.
    - SmoothGrad.
    - Blended Score-CAM + SmoothGrad.
# -----------------------------------------------------------------------------
This script generates:
    - CAM heatmaps.
    - Overlay images (X-ray & CAM).
    - Heatmap pattern classification (unilateral/bilateral/diffuse).
    - Severity estimation (mild/moderate/severe).
    - Normalised CAM outputs.
# -----------------------------------------------------------------------------
Note: This script is usee directly by the Phase 6 Streamlit dashboard to provide
clinically interpretable visualisations and structured inference outputs.
# -----------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# CNN architecture (Phase 5)
# --------------------------------------------------------------
class CNN_Train(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------
IMAGE_SIZE = 128

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --------------------------------------------------------------
# Load model
# --------------------------------------------------------------
def load_model(weights_path="P5_CNN_Model_WEIGHTS.pth"):
    """
    Load the trained Phase 5 CNN model for inference.

    Parameters:
        weights_path (str):
            Path to the saved model weights (.pth file). Defaults to the
            Phase 5 CNN weights used throughout the deployment pipeline.

    Returns:
        CNN_Train:
            The model loaded with pretrained weights, set to evaluation mode.

    """
    # Instatiate Phase 5 CNN architecture
    model = CNN_Train()
    
    # Load saved weights (CPU-compatible)
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Apply weights to model
    model.load_state_dict(state_dict)
    
    # Set model to inference mode
    model.eval()
    
    return model

# --------------------------------------------------------------
# Prediction
# --------------------------------------------------------------
def predict_image(model, pil_image):
    """
    Run inference on a single chest X-ray image using the Phase 5 CNN model.

    Steps:
        1. Convert PIL image to grayscale.
        2. Apply preprocessing (resize → tensor → normalise).
        3. Run forward pass through the CNN.
        4. Compute class probabilities using softmax.
        5. Determine predicted label (Normal vs Pneumonia).
        6. Compute confidence score based on predicted class.

    Parameters:
        model (CNN_Train):
            Loaded Phase 5 CNN model in evaluation mode.
        pil_image (PIL.Image):
            Input chest X-ray image.

    Returns:
        tuple:
            label (str): "NORMAL" or "PNEUMONIA"
            pneu_prob (float): Probability of pneumonia
            normal_prob (float): Probability of normal
            confidence (float): Probability of predicted class
            tensor (Tensor): Preprocessed image tensor (1, 128, 128)
    """
    # Convert image to grayscale 
    img = pil_image.convert("L")
    
    # Apply preprocessing pipeline and add batch dimension 
    tensor = test_transform(img).unsqueeze(0)
    
    # Disable gradients for inference
    with torch.no_grad():
        logits = model(tensor)                      # raw class scores
        probs = torch.softmax(logits, dim=1)[0]     # convert to probabilities
        
    # Extract class probabilities
    pneu_prob = probs[1].item()
    normal_prob = probs[0].item()
    
    # Determine predicted class
    label = "PNEUMONIA" if pneu_prob >= 0.5 else "NORMAL"
    
    # Confidence of predicted class
    confidence = pneu_prob if label == "PNEUMONIA" else normal_prob
    
    # Return squeezed tensor for CAM generation
    return label, pneu_prob, normal_prob, confidence, tensor.squeeze(0)

# --------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------
# - Global containers used by forward/backward hooks to store 
#   activations and gradients from the last convolutional layer.
# --------------------------------------------------------------
gradients = []
activations = []

def _forward_hook(module, inp, out):
    """
    Forward hook:
    - Captures the feature maps (activations) produced by the target
      convolutional layer during the forward pass.
    """
    activations.append(out)

def _backward_hook(module, grad_in, grad_out):
    """
    Backward hook:
    - Captures the gradients flowing back from the target class during
      backpropagation.
    - These gradients are used to weight the feature maps.
    """
    gradients.append(grad_out[0])

def generate_gradcam(model, img_tensor, target_class_idx=1):
    """
    Generate a Grad-CAM heatmap for a given image tensor.

    Steps:
        1. Register forward & backward hooks on the last convolutional layer.
        2. Run forward pass to obtain class probabilities.
        3. Backpropagate the score of the target class.
        4. Extract stored activations and gradients.
        5. Compute channel-wise weights by global average pooling.
        6. Combine weights with activations to produce CAM.
        7. Normalise CAM to [0, 1].
        8. Remove hooks and clear stored buffers.

    Parameters:
        model (nn.Module):
            The trained CNN model in evaluation mode.
        img_tensor (Tensor):
            Preprocessed image tensor of shape (1, 128, 128).
        target_class_idx (int):
            Index of the class to generate Grad-CAM for (default = 1 = Pneumonia).

    Returns:
        np.ndarray:
            Normalised Grad-CAM heatmap of shape (H, W).
    """
    # ensure inference mode
    model.eval()
    
    # Identify the last convolutional layer in the feature extractor
    last_conv = model.features[-5]
    
    # Register hooks to capture activations and gradients
    fh = last_conv.register_forward_hook(_forward_hook)
    bh = last_conv.register_full_backward_hook(_backward_hook)
    
    # Forward pass
    img_tensor = img_tensor.unsqueeze(0)           # add batch dimension
    scores = model(img_tensor)
    probs = torch.softmax(scores, dim=1)
    score = probs[0, target_class_idx]             # target class probability
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Retrieve stored activations and gradients
    acts =  activations[-1]
    grads = gradients[-1]
    
    # Compute channel weights using global average pooling over gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze().detach().numpy()
    
    # Weighted sum of activations implies raw CAM
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    
    # Clean-up hooks and buffers after Grad-CAM computation
    fh.remove()                     #  Remove forward hook from last conv layer
    bh.remove()                     #  Remove backward hook from last conv layer
    gradients.clear()              #  Clear stored gradients from global buffer
    activations.clear()            #  Clear stored activations from global buffer

    return cam

# --------------------------------------------------------------
# Score-CAM
# --------------------------------------------------------------
def generate_scorecam(model, img_tensor, target_class_idx=1):
    """
    Generate a Score-CAM heatmap for a given image tensor.

    Score-CAM works by:
        1. Extracting activation maps from the last convolutional layer.
        2. Upsampling each activation map to input size.
        3. Masking the input image with each activation map.
        4. Forward passing the masked image to measure class score change.
        5. Using score differences as weights for each activation map.
        6. Summing weighted activation maps → final CAM.
        7. Normalising CAM to [0, 1].

    Parameters:
        model (nn.Module):
            Trained CNN model in evaluation mode.
        img_tensor (Tensor):
            Preprocessed image tensor (1, 128, 128).
        target_class_idx (int):
            Class index to compute CAM for (default = 1 = Pneumonia).

    Returns:
        np.ndarray:
            Normalised Score-CAM heatmap of shape (H, W).
    """
    model.eval()                        # ensure inference mode
    
    # Identify the last convolutional layer
    last_conv = model.features[-5]
    
    # Temporary forward hook to capture activations
    def tmp_forward_hook(module, inp, out):
        tmp_forward_hook.activations = out
      
    # Register forward hook
    fh = last_conv.register_forward_hook(tmp_forward_hook)
    
    # Forward pass to compute baseline class score
    with torch.no_grad():
        img_batch = img_tensor.unsqueeze(0)
        scores = model(img_batch)
        probs = torch.softmax(scores, dim=1)
        base_score = probs[0, target_class_idx].item()
        
    # Retrieve activations from hook
    acts = tmp_forward_hook.activations.detach()
    fh.remove()                        # remove hook immediately after use
    
    # Remove batch dimension implies shape (C, H, W)
    acts = acts.squeeze(0)
    C, H, W = acts.shape
    
    # Initialise CAM accumulator
    cam = torch.zeros((H, W), dtype=torch.float32)
    
    # Add batch dimension back to input tensor
    img_tensor = img_tensor.unsqueeze(0)
    
    # Iterate over each activation map (channel)
    for c in range(C):
        
        # Extract activation map for channel c
        act_map = acts[c]
        
        # Normalise activation map to [0, 1]
        act_map_norm = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
        
        # Upsample activation map to match input image size
        act_map_resized = torch.nn.functional.interpolate(
            act_map_norm.unsqueeze(0).unsqueeze(0),
            size=img_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        ).squeeze()
        
        # Mask input image using resized activation map
        masked_input = img_tensor * act_map_resized.unsqueeze(0)
        
        # Forward pass on masked image
        with torch.no_grad():
            masked_scores = model(masked_input)
            masked_probs = torch.softmax(masked_scores, dim=1)
            masked_score = masked_probs[0, target_class_idx].item()
            
        # Weight = positive score difference
        weight = max(masked_score - base_score, 0.0)
        
        # Accumulate weighted activation map
        cam += weight * act_map
        
    # Convert to numpy and normalise
    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)

    return cam

# --------------------------------------------------------------
# SmoothGrad
# --------------------------------------------------------------
def generate_smoothgrad(model, img_tensor, target_class_idx=1, n_samples=32, noise_sigma=0.1):
    """
    Generate a SmoothGrad saliency map for a given image tensor.

    SmoothGrad works by:
        1. Adding random noise to the input image multiple times.
        2. Computing gradients of the target class score w.r.t. the input.
        3. Averaging gradients across all noisy samples.
        4. Taking absolute values and normalising → final CAM.

    Parameters:
        model (nn.Module):
            Trained CNN model in evaluation mode.
        img_tensor (Tensor):
            Preprocessed image tensor (1, 128, 128).
        target_class_idx (int):
            Class index to compute saliency for (default = 1 = Pneumonia).
        n_samples (int):
            Number of noisy samples to average over.
        noise_sigma (float):
            Standard deviation of Gaussian noise added to input.

    Returns:
        np.ndarray:
            Normalised SmoothGrad heatmap of shape (H, W).
    """
    model.eval()                     # ensure inference mode
    
    # Add batch dimension and enable gradient tracking
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    
    # Accumulator for gradients across noisy samples
    grads_accum = torch.zeros_like(img_tensor)
    
    # Generate multiple noisy samples
    for _ in range(n_samples):
        # Add Gaussian noise to input
        noise = torch.randn_like(img_tensor) * noise_sigma
        noisy_input = img_tensor + noise
        
        # Forward pass
        scores = model(noisy_input)
        probs = torch.softmax(scores, dim=1)
        score = probs[0, target_class_idx]
        
        # Backward pass
        model.zero_grad()
        score.backward(retain_graph=True)
        
        # Accumulate gradients
        grads_accum += img_tensor.grad
        
        # Reset gradients for next iteration
        img_tensor.grad.zero_()
        
    # Average gradients across samples
    grads_mean = grads_accum / n_samples
    
    # Convert to numpy and remove batch dimension
    cam = grads_mean.squeeze().detach().numpy()
    
    # Absolute value
    cam = np.abs(cam)
    
    # ReLU & normalisation
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)

    return cam

# --------------------------------------------------------------
# Blended Score-CAM + SmoothGrad (default)
# --------------------------------------------------------------
def generate_blended_cam(model, img_tensor, target_class_idx=1,
                         scorecam_weight=0.7, smoothgrad_weight=0.3):
    """
    Generate a blended CAM combining Score-CAM and SmoothGrad.

    Rationale:
        Score-CAM → spatially meaningful activation maps
        SmoothGrad → gradient-based fine-grained saliency

    Blending both improves interpretability by combining:
        - localisation (Score-CAM)
        - sensitivity (SmoothGrad)

    Parameters:
        model (nn.Module):
            Trained CNN model in evaluation mode.
        img_tensor (Tensor):
            Preprocessed image tensor (1, 128, 128).
        target_class_idx (int):
            Class index to compute CAM for (default = 1 = Pneumonia).
        scorecam_weight (float):
            Weight assigned to Score-CAM contribution.
        smoothgrad_weight (float):
            Weight assigned to SmoothGrad contribution.

    Returns:
        np.ndarray:
            Normalised blended CAM heatmap of shape (H, W).
    """
    # Generate Score-CAM and SmoothGrad maps
    score_cam = generate_scorecam(model, img_tensor, target_class_idx)
    smooth_cam = generate_smoothgrad(model, img_tensor, target_class_idx)

        # Resize Score-CAM to match SmoothGrad resolution
    score_cam_resized = Image.fromarray((score_cam * 255).astype(np.uint8)).resize(
        smooth_cam.shape[::-1]
    )
    score_cam_resized = np.array(score_cam_resized).astype(np.float32) / 255.0
    
    # Weighted blending of both CAMs
    blended = scorecam_weight * score_cam_resized + smoothgrad_weight * smooth_cam
    
    # ReLU + normalisation
    blended = np.maximum(blended, 0)
    blended /= (blended.max() + 1e-8)

    return blended

# --------------------------------------------------------------
# Overlay function
# --------------------------------------------------------------
def overlay_cam(pil_img, cam, gamma=0.5, threshold=0.20, intensity=0.65):
    """
    Create an overlay image by blending the CAM heatmap with the original X-ray.

    Steps:
        1. Convert CAM → uint8 → resize to match X-ray dimensions.
        2. Apply gamma correction to adjust contrast.
        3. Zero-out low activations using threshold.
        4. Convert CAM into a JET colour heatmap.
        5. Blend heatmap with original X-ray using intensity weighting.

    Parameters:
        pil_img (PIL.Image):
            Original chest X-ray image.
        cam (np.ndarray):
            Normalised CAM heatmap (H, W).
        gamma (float):
            Controls contrast of CAM (higher = more contrast).
        threshold (float):
            Minimum activation level to display (suppresses noise).
        intensity (float):
            Blend ratio between original image and heatmap.

    Returns:
        PIL.Image:
            Final overlay image (X-ray + CAM heatmap).
    """
    # Convert CAM to uint8 and resize to match original image size
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_resized = Image.fromarray(cam_uint8).resize(pil_img.size)
    cam_arr = np.array(cam_resized).astype(np.float32)
    
    # Apply gamma correction to adjust contrast
    cam_arr = np.power(cam_arr / 255.0, gamma)
    
    # Suppress low activations (noise removal)
    cam_arr[cam_arr < threshold] = 0.0
    
    # Convert CAM to JET colour map (RGB only, drop alpha)
    heatmap = plt.cm.jet(cam_arr)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Convert original X-ray to RGB array
    img_arr = np.array(pil_img.convert("RGB")).astype(np.float32)
    
    # Blend original image with heatmap using intensity weighting
    overlay = ((1 - intensity) * img_arr + intensity * heatmap).astype(np.uint8)

    return Image.fromarray(overlay)

# --------------------------------------------------------------
# Heatmap Pattern Classifier
# --------------------------------------------------------------
def classify_heatmap_pattern(cam, threshold=0.3):
    """
    Classify CAM activation pattern into one of four categories:
    • bilateral          → activation present in both lungs
    • unilateral_left    → activation present only in left lung region
    • unilateral_right   → activation present only in right lung region
    • diffuse            → no clear activation above threshold

    Parameters:
       cam (np.ndarray):
        Normalised CAM heatmap (H, W).
    threshold (float):
        Minimum activation level required to consider a pixel "active".

    Returns:
        str:
        One of: "bilateral", "unilateral_left", "unilateral_right", "diffuse".
    """
    # Normalise CAM to [0, 1]
    cam_norm = cam / (cam.max() + 1e-8)
    
    # Binary mask of activated pixels
    mask = cam_norm > threshold
    
    # Split mask into left and right halves
    mid = mask.shape[1] // 2
    
    # Total activation on left side
    left = mask[:, :mid].sum()
    
    # Total activation on right side
    right = mask[:, mid:].sum()
    
    # Classification logic
    if left > 0 and right > 0:
        return "Bilateral"
    elif left > 0:
        return "Unilateral_left"
    elif right > 0:
        return "Unilateral_right"
    else:
        return "Diffuse"

# --------------------------------------------------------------
# Severity Classifie
# --------------------------------------------------------------
def classify_severity(cam, label):  # CHANGED FROM 0.35 ON 23/09/2026 label ad
    """
    Estimate severity of activation based on mean CAM intensity.

    Rationale:
    - Higher CAM intensity → stronger model activation → potentially
      more extensive radiological abnormality.

    Severity thresholds:
    - Mild:     Implies mean activation < 0.15.
    - Moderate: Implies mean activation < 0.35.
    - Severe:   Implies mean activation ≥ 0.35.

    Parameters:
      cam (np.ndarray):
      Normalised CAM heatmap (H, W).

    Returns:
      str:
        One of: "mild", "moderate", "severe".
    """
    # If model predicts NORMAL
    if label == "NORMAL":
        return "Mild"

    cam_norm = cam / (cam.max() + 1e-8)
    mean_activation = np.percentile(cam_norm, 95) 
    
    # Threshold-based severity classification
    if mean_activation < 0.12:  
        return "Mild"
    elif mean_activation < 0.28:
        return "Moderate"
    else:
        return "Severe"

# --------------------------------------------------------------
# CAM Normalisation 
# --------------------------------------------------------------
def normalize_cam(cam):
    """
    Normalise a CAM heatmap to the range [0, 1].

    Steps:
        1. Apply ReLU (remove negative values).
        2. Divide by maximum activation to scale intensities.

    Parameters:
        cam (np.ndarray):
            Raw CAM heatmap (H, W), may contain negative values.

    Returns:
        np.ndarray:
            Normalised CAM heatmap in the range [0, 1].
    """
    # Remove negative values (ReLU)
    cam = np.maximum(cam, 0)
    # Scale to [0, 1] using max activation
    return cam / (cam.max() + 1e-8)