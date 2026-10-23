import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Crowd Density Estimation", page_icon="👥", layout="wide")
st.title("👥 CNN-Based Crowd Density Estimation")
st.caption("Pretrained YOLO person detection • Crowd count • Heatmap visualization")

@st.cache_resource
def load_model():
    # Downloads the pretrained YOLO11n weights automatically on first run.
    # Internet access is required the first time the app starts.
    return YOLO("yolo11n.pt")

def density_level(count):
    # Simple demo thresholds; adjust to your venue/image size.
    if count < 10:
        return "LOW", "🟢"
    elif count <= 25:
        return "MEDIUM", "🟠"
    return "HIGH", "🔴"

def make_heatmap(image_rgb, boxes_xyxy):
    """Creates a visual concentration map from detected person-box centers.
    This is a detection-based visualization, not a trained density-regression map.
    """
    h, w = image_rgb.shape[:2]
    points = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in boxes_xyxy:
        cx = int(max(0, min(w - 1, (x1 + x2) / 2)))
        cy = int(max(0, min(h - 1, (y1 + y2) / 2)))
        points[cy, cx] += 1.0

    # Spread each detection into a smooth hotspot for easier viewing.
    sigma = max(5, min(h, w) // 35)
    blurred = cv2.GaussianBlur(points, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if blurred.max() > 0:
        scaled = (blurred / blurred.max() * 255).astype(np.uint8)
        color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_rgb, 0.55, color, 0.45, 0)
    else:
        color = np.zeros_like(image_rgb)
        overlay = image_rgb.copy()
    return color, overlay

st.sidebar.header("Model settings")
confidence = st.sidebar.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
st.sidebar.info("The app uses a pretrained YOLO model to detect visible people. Dense/overlapping crowds may be undercounted.")

uploaded = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG crowd photo to analyze it.")
    st.markdown("**Features:** person detection, estimated count, heatmap overlay, and a simple crowd level.")
else:
    pil_image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(pil_image)

    with st.spinner("Loading CNN model and analyzing image..."):
        try:
            model = load_model()
            results = model.predict(source=image_rgb, conf=confidence, verbose=False)
            result = results[0]

            boxes_xyxy = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    # COCO class 0 is person.
                    if class_id == 0:
                        boxes_xyxy.append(box.xyxy[0].cpu().numpy().tolist())

            count = len(boxes_xyxy)
            level, emoji = density_level(count)

            annotated_bgr = result.plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            heatmap_rgb, overlay_rgb = make_heatmap(image_rgb, boxes_xyxy)

            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated people detected", count)
            c2.metric("Crowd level", f"{emoji} {level}")
            c3.metric("Detection confidence", f"{confidence:.2f}")

            tab1, tab2, tab3 = st.tabs(["Original image", "Person detections", "Heatmap overlay"])
            with tab1:
                st.image(image_rgb, caption="Uploaded crowd image", use_container_width=True)
            with tab2:
                st.image(annotated_rgb, caption="Detected people are marked with boxes", use_container_width=True)
            with tab3:
                st.image(overlay_rgb, caption="Detection-based crowd concentration visualization", use_container_width=True)

            png_buffer = BytesIO()
            Image.fromarray(annotated_rgb).save(png_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=png_buffer.getvalue(),
                file_name="crowd_detections.png",
                mime="image/png"
            )
            st.warning(
                "This count is an estimate from person detections, not a guaranteed exact count. "
                "Very dense crowds, small people, and occlusion can cause missed detections. "
                "The heatmap visualizes detection concentration; it is not a CSRNet-style density-regression output."
            )
        except Exception as e:
            st.error("Model loading or prediction failed.")
            st.exception(e)
            st.info("The first run needs internet access to download yolo11n.pt. Check Streamlit Cloud logs if deployment fails.")

st.divider()
st.caption("Final Year Project • Deep Learning & Computer Vision")
