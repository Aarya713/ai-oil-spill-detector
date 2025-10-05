import streamlit as st
import numpy as np
from PIL import Image
import requests
import tempfile
import os
import gdown
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection AI",
    page_icon="🌊",
    layout="wide"
)

class OilSpillDetector:
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.models_loaded = False
        self.load_models_from_drive()
    
    def load_models_from_drive(self):
        # Google Drive file IDs - YOU NEED TO UPDATE THESE!
        # Replace with your actual Google Drive file IDs
        SEGMENTATION_MODEL_ID = "YOUR_SEGMENTATION_MODEL_DRIVE_ID"
        CLASSIFICATION_MODEL_ID = "YOUR_CLASSIFICATION_MODEL_DRIVE_ID"
        
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Download segmentation model
            seg_model_path = "models/best_improved_unet_model.h5"
            if not os.path.exists(seg_model_path):
                with st.spinner("🔄 Downloading AI segmentation model..."):
                    gdown.download(f"https://drive.google.com/uc?id={https://drive.google.com/file/d/1oddiVJOirUYGhUnXHGqOrK8W1N7cUxIi/view?usp=drive_link}", seg_model_path, quiet=False)
            
            # Download classification model  
            cls_model_path = "models/simple_cnn_classifier.h5"
            if not os.path.exists(cls_model_path):
                with st.spinner("🔄 Downloading AI classification model..."):
                    gdown.download(f"https://drive.google.com/uc?id={https://drive.google.com/file/d/1EFYBAaMoLY6SCPO5fqyxPP870fnLbVtr/view?usp=drive_link}", cls_model_path, quiet=False)
            
            # Load models
            import tensorflow as tf
            if os.path.exists(seg_model_path):
                self.segmentation_model = tf.keras.models.load_model(seg_model_path, compile=False)
                st.sidebar.success("✅ Improved U-Net loaded")
            
            if os.path.exists(cls_model_path):
                self.classification_model = tf.keras.models.load_model(cls_model_path)
                st.sidebar.success("✅ CNN Classifier loaded")
            
            self.models_loaded = True
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Using demo mode: {str(e)}")
    
    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Resize to model input size
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((256, 256))
        image_resized = np.array(pil_image)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return np.expand_dims(image_normalized, axis=0)
    
    def predict_with_ai(self, image):
        if not self.models_loaded:
            return self.demo_predict(image)
        
        try:
            import tensorflow as tf
            processed_image = self.preprocess_image(image)
            
            # Use real AI models
            if self.classification_model:
                cls_pred = self.classification_model.predict(processed_image, verbose=0)[0][0]
                classification_has_oil = cls_pred > 0.5
                classification_confidence = float(cls_pred)
            else:
                classification_has_oil = False
                classification_confidence = 0.5
            
            if self.segmentation_model:
                seg_pred = self.segmentation_model.predict(processed_image, verbose=0)
                
                # Process segmentation output
                if len(seg_pred[0].shape) == 3:
                    pred_single = seg_pred[0][:, :, 0]
                else:
                    pred_single = seg_pred[0]
                
                binary_mask = (pred_single > 0.5).astype(np.uint8)
                oil_pixels = np.sum(binary_mask)
                oil_percentage = (oil_pixels / binary_mask.size) * 100
                
                segmentation_has_oil = oil_percentage > 0.5
                segmentation_confidence = min(0.95, 0.7 + (oil_percentage / 100) * 0.25)
                
                # Combine results
                has_oil = classification_has_oil or segmentation_has_oil
                confidence = (classification_confidence * 0.3 + segmentation_confidence * 0.7) * 100
                
                return has_oil, binary_mask, confidence, oil_percentage
            else:
                return self.demo_predict(image)
                
        except Exception as e:
            st.error(f"AI prediction error: {str(e)}")
            return self.demo_predict(image)
    
    def demo_predict(self, image):
        # Fallback to demo mode
        width, height = 256, 256
        mask = self.create_demo_mask()
        oil_pixels = np.sum(mask)
        oil_percentage = (oil_pixels / mask.size) * 100
        has_oil = oil_percentage > 5
        confidence = min(95.0, 70 + oil_percentage * 0.5)
        return has_oil, mask, confidence, oil_percentage
    
    def create_demo_mask(self):
        # Create demo mask using PIL
        from PIL import ImageDraw
        mask = Image.new('L', (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw oil spill shapes
        draw.ellipse([68, 88, 188, 168], fill=255)  # Main spill
        draw.ellipse([60, 60, 100, 100], fill=255)   # Small spill 1
        draw.ellipse([150, 140, 190, 180], fill=255) # Small spill 2
        
        return np.array(mask)

def create_visualization(original_image, segmentation_mask):
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image
        
    if original_np.shape[-1] == 4:
        original_np = original_np[..., :3]

    # Resize for display
    display_size = (400, 400)
    pil_original = Image.fromarray(original_np)
    display_original = pil_original.resize(display_size)
    
    # Create colored mask
    pil_mask = Image.fromarray(segmentation_mask).resize(display_size, Image.NEAREST)
    colored_mask = Image.new('RGB', display_size, (0, 0, 0))
    mask_array = np.array(pil_mask)
    colored_array = np.array(colored_mask)
    colored_array[mask_array > 0] = [255, 0, 124]  # Pink for oil
    colored_mask = Image.fromarray(colored_array)
    
    # Create overlay
    overlay = display_original.copy()
    overlay_array = np.array(overlay).astype(float)
    mask_resized_array = np.array(pil_mask)
    oil_areas = mask_resized_array > 0
    
    overlay_array[oil_areas] = overlay_array[oil_areas] * 0.6 + np.array([255, 0, 124]) * 0.4
    overlay_array = np.clip(overlay_array, 0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_array)
    
    return display_original, colored_mask, overlay

def main():
    st.title("🌊 AI Oil Spill Detection System")
    
    # Initialize detector (will download models from Google Drive)
    detector = OilSpillDetector()
    
    st.sidebar.title("📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose satellite image", type=["jpg", "jpeg", "png"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Model Status:** " + ("✅ AI Models Loaded" if detector.models_loaded else "🔶 Demo Mode"))
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
        if st.button("🔍 Detect Oil Spills", type="primary"):
            with st.spinner("🛰️ AI analyzing image..."):
                # Use real AI models or demo
                has_oil, mask, confidence, oil_percent = detector.predict_with_ai(image)
                
                # Display results
                st.subheader("📊 Detection Results")
                if has_oil:
                    st.error("🚨 OIL SPILL DETECTED")
                    st.write(f"**Using:** {'Real AI Models' if detector.models_loaded else 'Demo Mode'}")
                else:
                    st.success("✅ NO OIL DETECTED")
                    st.write(f"**Using:** {'Real AI Models' if detector.models_loaded else 'Demo Mode'}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{confidence:.1f}%")
                with col2:
                    st.metric("Oil Coverage", f"{oil_percent:.2f}%")
                with col3:
                    risk = "HIGH" if oil_percent > 15 else "MEDIUM" if oil_percent > 5 else "LOW"
                    st.metric("Risk Level", risk)
                
                # Show visualizations
                st.subheader("🔍 AI Analysis")
                orig_viz, mask_viz, overlay_viz = create_visualization(image, mask)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(orig_viz, caption="Original", use_container_width=True)
                with col2:
                    st.image(mask_viz, caption="AI Detection", use_container_width=True)
                with col3:
                    st.image(overlay_viz, caption="Overlay", use_container_width=True)
    
    else:
        st.info("👆 Upload a satellite image to start AI detection")
        st.markdown("### 🌟 Features:")
        st.markdown("- **Improved U-Net Segmentation**")
        st.markdown("- **CNN Classification**")
        st.markdown("- **Real AI Models from Google Drive**")
        st.markdown("- **Automated Risk Assessment**")

if __name__ == "__main__":
    main()
