import streamlit as st
import cv2
import numpy as np
import zipfile
import io

# --- Page config ---
st.set_page_config(page_title=" Image Transformer", layout="wide")

# --- Custom CSS for colors and styling ---
custom_css = """
<style>
    /* Hide default menu, footer, header */
    #MainMenu, footer, header {visibility: hidden;}
    /* Page background */
    .stApp {
        background-color: #f0f4f8;
        color: #222222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Sidebar background and text */
    [data-testid="stSidebar"] {
        background-color: #1E3A20;
        color: #a6d785;
        font-weight: 600;
    }
    /* Sidebar title color */
    .sidebar .sidebar-content > div:first-child h2 {
        color: #a6d785 !important;
        font-size: 24px;
        font-weight: 700;
    }
    /* Headers on main page */
    h1, h2, h3, h4 {
        color: #2a7a33;
        font-weight: 700;
    }
    /* Download button color */
    div.stDownloadButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 20px;
        border: none;
    }
    div.stDownloadButton > button:hover {
        background-color: #388E3C;
        color: white;
    }
    /* Sliders and Selectbox accent color */
    .stSlider > div > div > div > div[role="slider"] {
        background-color: #4CAF50 !important;
    }
    .stSelectbox > div > div > div > div[role="listbox"] {
        border-color: #4CAF50 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar title
st.sidebar.title("📂 Upload Your Data")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload image here", type=['jpg', 'jpeg', 'png'])

transformed_images = {}

if uploaded_file is None:
    st.markdown(
    "<span style='color: black; font-weight: bold;'> Please upload an image to begin...</span>",
    unsafe_allow_html=True
)
else:
    img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, 1)

    if img_bgr is None:
        st.error(" Unsupported or corrupted image format.")
    else:
        width = 600
        height = int(img_bgr.shape[0] * (600 / img_bgr.shape[1]))
        img_resized = cv2.resize(img_bgr, (width, height))
        
        st.subheader(" Original Image Preview (Full Resolution)")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=False)
        st.markdown("----")

        format_options = [
            "Grayscale", "Rotate", "Flip Horizontally", "Flip Vertically",
            "Shearing", "Translation", "Cropping", "Edge Detection",
            "HSV Conversion", "LAB Conversion", "YCrCb Conversion"
        ]
        selected_formats = st.sidebar.multiselect("Choose transformations", format_options, default=[])

        st.subheader(" Transformed Outputs")

        for i in range(0, len(selected_formats), 2):
            col_pair = st.columns(2)
            for j in range(2):
                if i + j < len(selected_formats):
                    fmt = selected_formats[i + j]
                    result = None
                    with col_pair[j]:
                        st.markdown(f"**{fmt}**")

                        if fmt == 'Grayscale':
                            result = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                            st.image(result, use_container_width=True, clamp=True)

                        elif fmt == "Rotate":
                            rotation_option = st.sidebar.selectbox(
                                "Select Rotation",
                                ["No Rotation", "90° Clockwise", "180°", "90° Counter-Clockwise"],
                                key="rotate"
                            )
                            rot_map = {
                                "90° Clockwise": cv2.ROTATE_90_CLOCKWISE,
                                "180°": cv2.ROTATE_180,
                                "90° Counter-Clockwise": cv2.ROTATE_90_COUNTERCLOCKWISE
                            }
                            if rotation_option != "No Rotation":
                                result = cv2.rotate(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), rot_map[rotation_option])
                            else:
                                result = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                            st.image(result, use_container_width=True, clamp=True)

                        elif fmt == "Flip Horizontally":
                            flip_img = cv2.flip(img_resized, 1)
                            result = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
                            st.image(result, use_container_width=True, clamp=True)

                        elif fmt == "Flip Vertically":
                            flip_img = cv2.flip(img_resized, 0)
                            result = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
                            st.image(result, use_container_width=True, clamp=True)

                        elif fmt == 'Shearing':
                            shear = st.sidebar.slider("Shear Factor", 0.0, 1.0, 0.2, step=0.05)
                            rows, cols = img_resized.shape[:2]
                            shear_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
                            new_width = int(cols + shear * rows)
                            shear_img = cv2.warpAffine(img_resized, shear_matrix, (new_width, rows))
                            result = cv2.cvtColor(shear_img, cv2.COLOR_BGR2RGB)
                            st.image(result, caption=f"Sheared Image (factor: {shear})", use_container_width=True, clamp=True)

                        elif fmt == "Translation":
                            rows, cols = img_resized.shape[:2]
                            col1, col2 = st.sidebar.columns(2)
                            with col1:
                                x_shift = st.number_input("X-shift", -cols, cols, 0, key="tx")
                            with col2:
                                y_shift = st.number_input("Y-shift", -rows, rows, 0, key="ty")
                            m = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                            tras_img = cv2.warpAffine(img_resized, m, (cols, rows))
                            result = cv2.cvtColor(tras_img, cv2.COLOR_BGR2RGB)
                            st.image(result, use_container_width=True, clamp=True)

                        elif fmt == "Cropping":
                            rows, cols = img_resized.shape[:2]
                            st.sidebar.markdown("###  Crop Parameters")
                            x_start = st.sidebar.slider("x_start", 0, cols - 1, 0)
                            x_end = st.sidebar.slider("x_end", x_start + 1, cols, cols)
                            y_start = st.sidebar.slider("y_start", 0, rows - 1, 0)
                            y_end = st.sidebar.slider("y_end", y_start + 1, rows, rows)
                            crop_img = img_resized[y_start:y_end, x_start:x_end]
                            result = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                            st.image(result, use_container_width=True, clamp=True)

                        elif fmt == "Edge Detection":
                            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 100, 200)
                            result = edges
                            st.image(result, caption="Canny Edge Detection", use_container_width=True, clamp=True)

                        elif fmt == "HSV Conversion":
                            hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv_img)
                            st.image(h, caption="Hue Channel", use_container_width=True, clamp=True)
                            st.image(s, caption="Saturation Channel", use_container_width=True, clamp=True)
                            st.image(v, caption="Value Channel", use_container_width=True, clamp=True)
                            result = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

                        elif fmt == "LAB Conversion":
                            lab_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab_img)
                            st.image(l, caption="L (Lightness) Channel", use_container_width=True, clamp=True)
                            st.image(a, caption="A (Green-Red) Channel", use_container_width=True, clamp=True)
                            st.image(b, caption="B (Blue-Yellow) Channel", use_container_width=True, clamp=True)
                            result = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

                        elif fmt == "YCrCb Conversion":
                            ycrcb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
                            y, cr, cb = cv2.split(ycrcb_img)
                            st.image(y, caption="Y (Luma) Channel", use_container_width=True, clamp=True)
                            st.image(cr, caption="Cr (Red Diff.) Channel", use_container_width=True, clamp=True)
                            st.image(cb, caption="Cb (Blue Diff.) Channel", use_container_width=True, clamp=True)
                            result = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

                        # Save to dictionary
                        if result is not None:
                            if len(result.shape) == 2:
                                rgb_img = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
                            else:
                                rgb_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            success, buffer = cv2.imencode('.jpg', rgb_img)
                            if success:
                                transformed_images[f"{fmt.lower().replace(' ', '_')}.jpg"] = buffer.tobytes()

        if transformed_images:
            st.markdown("###  Download All Transformed Images")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                for filename, image_bytes in transformed_images.items():
                    zf.writestr(filename, image_bytes)
            zip_buffer.seek(0)
            st.download_button(
                label=" Download ZIP",
                data=zip_buffer,
                file_name="transformed_images.zip",
                mime="application/zip"
            )
