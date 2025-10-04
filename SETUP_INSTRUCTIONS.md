# Google Drive Model Setup Guide

## Step 1: Get Your Google Drive File IDs

1. Upload your models to Google Drive:
   - Go to https://drive.google.com/
   - Upload best_unet_model.h5 and simple_cnn_classifier.h5

2. Get file IDs:
   - Right-click each file -> "Get link"
   - Make sure link sharing is "Anyone with the link"
   - Copy the file ID from the URL

## Step 2: Update the App Code

In oil_spill_app.py, find these lines and replace with your actual file IDs:

'classification': {
    'drive_id': 'YOUR_ACTUAL_CLASSIFICATION_MODEL_ID',
},
'segmentation': {
    'drive_id': 'YOUR_ACTUAL_SEGMENTATION_MODEL_ID',
}

## Step 3: Deploy on Streamlit

1. Upload all files to GitHub
2. Deploy on Streamlit Cloud
3. Models will auto-download on first use
