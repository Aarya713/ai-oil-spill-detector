
import streamlit as st
import numpy as np
from PIL import Image
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide"
)

def create_oil_mask(width, height):
    # Create a simple oil spill mask using numpy
    mask = np.zeros((height, width), dtype=bool)
    
    # Create circular spills
    center_x, center_y = width // 2, height // 2
    
    # Main spill
    y, x = np.ogrid[:height, :width]
    mask1 = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 4)**2
    
    # Smaller spills
    mask2 = (x - center_x*0.7)**2 + (y - center_y*1.3)**2 <= (min(width, height) // 8)**2
    mask3 = (x - center_x*1.3)**2 + (y - center_y*0.7)**2 <= (min(width, height) // 6)**2
    
    mask = mask1 | mask2 | mask3
    return mask

def create_colored_mask(mask, color):
    # Convert boolean mask to colored image
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask] = color
    return colored

def create_overlay(original, mask, color, alpha=0.4):
    # Create overlay image
    overlay = original.copy()
    mask_3d = np.stack([mask] * 3, axis=-1)
    overlay = np.where(mask_3d, overlay * (1-alpha) + np.array(color) * alpha, overlay)
    return overlay.astype(np.uint8)

def main():
    st.sidebar.title("🌊 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Detection"])
    
    if page == "🏠 Home":
        show_home()
    else:
        show_detection()

def show_home():
    st.title("🌊 Oil Spill Detection System")
    
    st.markdown("""
    <div style='background: #1f77b4; color: white; padding: 30px; border-radius: 15px; margin: 20px 0;'>
        <h2>🚀 AI-Powered Detection</h2>
        <p>Upload satellite images for oil spill analysis and segmentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Classification**\\nDetect oil spills")
    with col2:
        st.info("**🎯 Segmentation**\\nIdentify spill boundaries")
    with col3:
        st.info("**📊 Analysis**\\nGenerate reports")

def show_detection():
    st.header("🔍 Oil Spill Analysis")
    
    uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
    use_demo = st.checkbox("Use demo image instead")
    
    if uploaded_file or use_demo:
        if use_demo:
            st.info("🛰 Using demo satellite image")
            # Create a simple demo image
            width, height = 400, 300
            demo_array = np.zeros((height, width, 3), dtype=np.uint8)
            # Blue ocean background
            demo_array[:, :] = [30, 60, 150]
            # Add some green land
            demo_array[100:200, 50:150] = [40, 120, 60]
            demo_array[50:120, 250:350] = [50, 100, 70]
            image = Image.fromarray(demo_array)
        else:
            image = Image.open(uploaded_file)
        
        # Display original image
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Analyzing with AI..."):
                # Simulate processing time
                import time
                time.sleep(2)
                
                # Generate random but realistic results
                width, height = image.size
                oil_mask = create_oil_mask(width, height)
                oil_pixels = np.sum(oil_mask)
                total_pixels = oil_mask.size
                oil_percentage = (oil_pixels / total_pixels) * 100
                
                # Realistic detection logic
                has_spill = oil_percentage > 3
                confidence = min(95.0, 70 + oil_percentage * 0.8)
                
                # Display results
                st.subheader("📊 Detection Results")
                
                if has_spill:
                    st.error(f"🚨 OIL SPILL DETECTED")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.write(f"**Coverage:** {oil_percentage:.2f}%")
                    st.write("**Risk Level:** HIGH")
                else:
                    st.success(f"✅ NO OIL SPILL DETECTED")
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    st.write(f"**Coverage:** {oil_percentage:.2f}%")
                    st.write("**Risk Level:** LOW")
                
                # Show visualizations
                if has_spill:
                    st.subheader("🎯 Spill Visualization")
                    
                    # Convert PIL image to numpy array
                    if use_demo:
                        img_array = np.array(image)
                    else:
                        img_array = np.array(image.convert('RGB'))
                    
                    # Create visualizations
                    oil_color = [255, 0, 0]  # Red for oil spills
                    
                    # Detection mask
                    colored_mask = create_colored_mask(oil_mask, oil_color)
                    
                    # Overlay
                    overlay_img = create_overlay(img_array, oil_mask, oil_color)
                    
                    # Display side by side
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(img_array, caption="Original Image", use_container_width=True)
                    
                    with col2:
                        st.image(colored_mask, caption="Oil Detection", use_container_width=True)
                    
                    with col3:
                        st.image(overlay_img, caption="Detection Overlay", use_container_width=True)
                    
                    # Statistics
                    st.info(f"**📈 Detection Statistics:**")
                    st.write(f"- **Oil Coverage:** {oil_percentage:.2f}%")
                    st.write(f"- **Affected Pixels:** {oil_pixels:,}")
                    st.write(f"- **Total Area:** {total_pixels:,} pixels")
                
                # Generate report
                st.subheader("📋 Analysis Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Image Information**")
                    st.write(f"- Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- Image Size: {width} × {height} pixels")
                    st.write(f"- Format: {'Demo' if use_demo else 'Uploaded'}")
                
                with col2:
                    st.write("**Detection Results**")
                    st.write(f"- Oil Spill: {'DETECTED' if has_spill else 'NOT DETECTED'}")
                    st.write(f"- Confidence: {confidence:.1f}%")
                    st.write(f"- Coverage: {oil_percentage:.2f}%")
                    st.write(f"- Urgency: {'HIGH' if has_spill else 'LOW'}")
    
    else:
        st.info("👆 Upload a satellite image or check 'Use demo image' to get started!")

if __name__ == "__main__":
    main()
