# Oil Spill Detection System

AI-powered web app that detects oil spills in satellite imagery.

## Features
- Upload satellite images
- AI-powered oil spill detection
- Visual results with overlays
- Risk assessment
- Google Drive model integration

## Setup
1. Get Google Drive file IDs for your models
2. Update the drive_id values in oil_spill_app.py
3. Deploy on Streamlit Cloud

## Models Used
- best_unet_model.h5 - Segmentation
- simple_cnn_classifier.h5 - Classification
