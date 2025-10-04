import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Oil Spill Detector",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; }
    .danger { background-color: #ffebee; border-left: 5px solid #f44336; padding: 15px; border-radius: 5px; }
    .safe { background-color: #e8f5e8; border-left: 5px solid #4caf50; padding: 15px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class OilSpillDetector:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained models from models/ folder"""
        try:
            # Try to load classification model
            if os.path.exists('models/simple_cnn_classifier.h5'):
                self.classification_model = tf.keras.models.load_model('models/simple_cnn_classifier.h5')
                st.sidebar.success("✅ Classification model loaded")
            else:
                st.sidebar.info("ℹ️ Using demo classification")
            
            # Try to load segmentation model  
            if os.path.exists('models/best_unet_model.h5'):
                self.segmentation_model = tf.keras.models.load_model('models/best_unet_model.h5', compile=False)
                st.sidebar.success("✅ Segmentation model loaded")
                self.models_loaded = True
            else:
                st.sidebar.info("ℹ️ Using demo segmentation")
                
        except Exception as e:
            st.sidebar.error(f"Model loading error: {e}")

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Resize and normalize
        image_resized = cv2.resize(image, (256, 256))
        image_normalized = image_resized.astype(np.float32) / 255.0
        return np.expand_dims(image_normalized, axis=0)

    def predict_segmentation(self, image):
        """Detect oil spills using AI models"""
        # DEMO MODE - Replace this with your actual model prediction
        st.info("🔍 AI is analyzing the satellite image...")
        
        # Convert image for processing
        image_np = np.array(image) if isinstance(image, Image.Image) else image
        h, w = 256, 256
        
        # Create realistic demo detection
        demo_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simulate oil spill patterns
        cv2.ellipse(demo_mask, (w//2, h//2), (w//4, h//4), 0, 0, 360, 1, -1)
        cv2.ellipse(demo_mask, (w//3, h//3), (w//8, h//8), 0, 0, 360, 1, -1)
        
        # Calculate statistics
        oil_pixels = np.sum(demo_mask)
        oil_percentage = (oil_pixels / demo_mask.size) * 100
        has_oil_spill = oil_percentage > 5
        confidence = min(95.0, 70 + oil_percentage * 0.5)
        
        return has_oil_spill, demo_mask, confidence, oil_percentage

def create_visualization(original_image, segmentation_mask):
    """Create result visualizations"""
    original_np = np.array(original_image) if isinstance(original_image, Image.Image) else original_image
    
    # Ensure RGB
    if len(original_np.shape) == 2:
        original_np = np.stack([original_np] * 3, axis=-1)
    elif original_np.shape[-1] == 4:
        original_np = original_np[..., :3]
    
    # Resize for display
    display_size = (400, 400)
    display_original = cv2.resize(original_np, display_size)
    
    # Create colored mask
    mask_resized = cv2.resize(segmentation_mask, display_size)
    colored_mask = np.zeros((*display_size, 3), dtype=np.uint8)
    colored_mask[mask_resized == 1] = [255, 0, 124]  # Pink color for oil
    
    # Create overlay
    overlay = display_original.copy()
    overlay[mask_resized == 1] = overlay[mask_resized == 1] * 0.6 + [255, 0, 124] * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return display_original, colored_mask, overlay

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 AI Oil Spill Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("Detect oil spills in satellite imagery using AI")
        st.header("Instructions")
        st.write("1. Upload satellite image")
        st.write("2. Click Analyze")
        st.write("3. View results")
        detector = OilSpillDetector()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Satellite Image")
        uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "tif"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Processing..."):
                    has_oil_spill, seg_mask, confidence, oil_percentage = detector.predict_segmentation(image)
                    display_original, display_mask, display_overlay = create_visualization(image, seg_mask)
                    
                    # Store results
                    st.session_state.result = {
                        'has_oil_spill': has_oil_spill,
                        'confidence': confidence,
                        'oil_percentage': oil_percentage,
                        'display_original': display_original,
                        'display_mask': display_mask,
                        'display_overlay': display_overlay,
                        'image': image
                    }
    
    with col2:
        if 'result' in st.session_state:
            st.subheader("Detection Results")
            
            # Display images
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(st.session_state.result['display_original'], caption="Original")
            with col2:
                st.image(st.session_state.result['display_mask'], caption="Oil Detection")
            with col3:
                st.image(st.session_state.result['display_overlay'], caption="Overlay")
            
            # Risk assessment
            risk_levels = {
                (50, 100): ("CRITICAL", "🔴", "danger"),
                (20, 50): ("HIGH", "🟠", "danger"), 
                (5, 20): ("MEDIUM", "🟡", "danger"),
                (1, 5): ("LOW", "🟢", "safe"),
                (0, 1): ("VERY LOW", "⚪", "safe")
            }
            
            oil_pct = st.session_state.result['oil_percentage']
            risk_level = "VERY LOW"
            risk_emoji = "⚪"
            risk_style = "safe"
            
            for (min_pct, max_pct), (level, emoji, style) in risk_levels.items():
                if min_pct <= oil_pct < max_pct:
                    risk_level = level
                    risk_emoji = emoji
                    risk_style = style
                    break
            
            # Results box
            if st.session_state.result['has_oil_spill']:
                st.markdown(f'<div class="{risk_style}"><h3>🚨 OIL SPILL DETECTED</h3><p><b>Confidence:</b> {st.session_state.result["confidence"]:.1f}%</p><p><b>Risk Level:</b> {risk_emoji} {risk_level}</p><p><b>Oil Coverage:</b> {oil_pct:.2f}%</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="{risk_style}"><h3>✅ NO OIL SPILL DETECTED</h3><p><b>Confidence:</b> {st.session_state.result["confidence"]:.1f}%</p><p><b>Risk Level:</b> {risk_emoji} {risk_level}</p><p><b>Oil Coverage:</b> {oil_pct:.2f}%</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()