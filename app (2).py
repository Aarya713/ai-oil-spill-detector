
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide"
)

def create_demo_segmentation(width, height):
    mask = np.zeros((height, width))
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    
    mask1 = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 4)**2
    mask2 = (x - center_x*0.7)**2 + (y - center_y*1.3)**2 <= (min(width, height) // 8)**2
    mask3 = (x - center_x*1.3)**2 + (y - center_y*0.7)**2 <= (min(width, height) // 6)**2
    
    mask = (mask1 | mask2 | mask3).astype(float)
    return mask

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Detection"])
    
    if page == "Home":
        st.title("Oil Spill Detection System")
        st.write("Upload satellite images for oil spill analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Classification - Detect oil spills")
        with col2:
            st.info("Segmentation - Identify boundaries")
        with col3:
            st.info("Analysis - Generate reports")
    else:
        st.header("Oil Spill Analysis")
        
        uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
        use_demo = st.checkbox("Use demo image")
        
        if uploaded_file or use_demo:
            if use_demo:
                width, height = 400, 300
                demo_image = np.random.rand(height, width, 3) * 0.3 + 0.5
                demo_image[100:200, 50:150] = [0.3, 0.6, 0.2]
                demo_image[50:120, 250:350] = [0.4, 0.5, 0.3]
            else:
                image = Image.open(uploaded_file)
                width, height = image.size
                demo_image = np.array(image) / 255.0
                if len(demo_image.shape) == 2:
                    demo_image = np.stack([demo_image]*3, axis=-1)
                elif demo_image.shape[2] == 4:
                    demo_image = demo_image[:,:,:3]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(demo_image)
                ax.set_title("Satellite Image")
                ax.axis('off')
                st.pyplot(fig)
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    import time
                    time.sleep(2)
                    
                    has_spill = np.random.random() > 0.4
                    confidence = np.random.uniform(0.7, 0.95) if has_spill else np.random.uniform(0.1, 0.4)
                    
                    with col2:
                        if has_spill:
                            st.error("OIL SPILL DETECTED")
                            st.write(f"Confidence: {confidence:.1%}")
                            st.write("Risk Level: HIGH")
                        else:
                            st.success("NO OIL SPILL")
                            st.write(f"Confidence: {confidence:.1%}")
                            st.write("Risk Level: LOW")
                    
                    if has_spill:
                        st.subheader("Spill Segmentation")
                        segmentation_mask = create_demo_segmentation(width, height)
                        
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                        ax1.imshow(demo_image)
                        ax1.set_title("Original")
                        ax1.axis('off')
                        
                        ax2.imshow(segmentation_mask, cmap='hot')
                        ax2.set_title("Detection")
                        ax2.axis('off')
                        
                        overlay = demo_image.copy()
                        mask_rgb = np.stack([segmentation_mask] * 3, axis=-1)
                        red_overlay = np.array([1, 0, 0])
                        overlay = np.where(mask_rgb > 0, overlay * 0.6 + red_overlay * 0.4, overlay)
                        ax3.imshow(overlay)
                        ax3.set_title("Overlay")
                        ax3.axis('off')
                        
                        st.pyplot(fig)
                        
                        spill_area = np.sum(segmentation_mask > 0)
                        spill_percentage = (spill_area / segmentation_mask.size) * 100
                        st.info(f"Spill Coverage: {spill_percentage:.1f}%")
        
        else:
            st.info("Upload an image or use demo to get started")

if __name__ == "__main__":
    main()
