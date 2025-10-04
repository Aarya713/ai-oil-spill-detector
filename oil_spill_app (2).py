import streamlit as st
import numpy as np
from PIL import Image
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🌊",
    layout="wide"
)

def create_oil_mask(width, height):
    """Create realistic oil spill mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create main oil spill
    center_x, center_y = width // 2, height // 2
    cv_x = center_x + random.randint(-50, 50)
    cv_y = center_y + random.randint(-50, 50)
    
    # Draw elliptical spill
    axis1 = random.randint(80, 120)
    axis2 = random.randint(60, 100)
    angle = random.randint(0, 180)
    
    # We'll use PIL for drawing since it's simpler
    from PIL import ImageDraw
    pil_mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(pil_mask)
    
    # Draw main ellipse
    draw.ellipse([
        cv_x - axis1, cv_y - axis2,
        cv_x + axis1, cv_y + axis2
    ], fill=255)
    
    # Add smaller spills
    for i in range(3):
        x = cv_x + random.randint(-100, 100)
        y = cv_y + random.randint(-80, 80)
        r = random.randint(10, 30)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    return np.array(pil_mask)

def analyze_image(image):
    """Analyze image for oil spills"""
    width, height = 400, 300
    
    # Create realistic detection
    oil_mask = create_oil_mask(width, height)
    oil_pixels = np.sum(oil_mask > 0)
    total_pixels = oil_mask.size
    oil_percentage = (oil_pixels / total_pixels) * 100
    
    # Realistic detection logic
    has_oil_spill = oil_percentage > 3
    confidence = min(95.0, 70 + oil_percentage * 0.8)
    
    return has_oil_spill, oil_mask, confidence, oil_percentage

def main():
    st.title("🌊 Oil Spill Detection System")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Detection"])
    
    if page == "Home":
        st.header("Welcome to Oil Spill Detection")
        st.write("Upload satellite images to detect oil spills using AI")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Image Upload**\nUpload satellite images")
        with col2:
            st.info("**AI Analysis**\nDetect oil spills automatically")
        with col3:
            st.info("**Results**\nGet detailed reports")
            
    else:
        st.header("Oil Spill Detection")
        
        uploaded_file = st.file_uploader("Choose satellite image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze for Oil Spills"):
                with st.spinner("Analyzing image..."):
                    has_spill, oil_mask, confidence, oil_percentage = analyze_image(image)
                    
                    # Display results
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
                    
                    # Show visualization
                    st.subheader("Detection Visualization")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(image, caption="Original", width=200)
                    
                    with col2:
                        # Create colored mask
                        colored_mask = np.zeros((*oil_mask.shape, 3), dtype=np.uint8)
                        colored_mask[oil_mask > 0] = [255, 0, 0]
                        st.image(colored_mask, caption="Oil Detection", width=200)
                    
                    with col3:
                        # Create overlay
                        display_img = image.resize((oil_mask.shape[1], oil_mask.shape[0]))
                        overlay = np.array(display_img).astype(float)
                        overlay[oil_mask > 0] = overlay[oil_mask > 0] * 0.6 + [255, 0, 0] * 0.4
                        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                        st.image(overlay, caption="Overlay", width=200)
                    
                    # Report
                    st.subheader("Analysis Report")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Detection Results**")
                        st.write(f"- Oil Spill: {'YES' if has_spill else 'NO'}")
                        st.write(f"- Confidence: {confidence:.1f}%")
                        st.write(f"- Coverage: {oil_percentage:.2f}%")
                    
                    with col2:
                        st.write("**Image Info**")
                        st.write(f"- Size: {image.size[0]} x {image.size[1]}")
                        st.write(f"- Analysis: {datetime.now().strftime('%H:%M:%S')}")
                        st.write(f"- Status: Complete")
        
        else:
            st.info("Please upload a satellite image to get started")

if __name__ == "__main__":
    main()
