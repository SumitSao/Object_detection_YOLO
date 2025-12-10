"""
YOLO Object Detection Application with Authentication
Main Streamlit application integrating login/signup with object detection features.
"""

import streamlit as st
from auth import AuthUI
import os
from PIL import Image
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np


# Page configuration
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_model():
    """Load YOLO model."""
    model_path = "yolo11n.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model file '{model_path}' not found!")
        return None


def detect_objects_image(model, image, conf_threshold=0.25):
    """
    Perform object detection on an image.
    
    Args:
        model: YOLO model
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold for detection
        
    Returns:
        Annotated image with detections
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run inference
    results = model(image, conf=conf_threshold)
    
    # Get annotated image
    annotated_image = results[0].plot()
    
    return annotated_image, results[0]


def detect_objects_video(model, video_path, conf_threshold=0.25):
    """
    Perform object detection on a video.
    
    Args:
        model: YOLO model
        video_path: Path to video file
        conf_threshold: Confidence threshold for detection
        
    Returns:
        Path to output video
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    progress_bar.empty()
    
    return output_path


def show_image_detection_page(model):
    """Display image detection interface."""
    st.header("📷 Image Object Detection")
    st.markdown("Upload an image to detect objects using YOLO.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect Objects", use_container_width=True):
                with st.spinner("Detecting objects..."):
                    annotated_image, results = detect_objects_image(model, image, conf_threshold)
                    
                    # Store results in session state
                    st.session_state.detection_result = annotated_image
                    st.session_state.detection_info = results
    
    with col2:
        if 'detection_result' in st.session_state:
            st.subheader("Detection Results")
            st.image(st.session_state.detection_result, caption="Detected Objects", use_container_width=True)
            
            # Show detection statistics
            if st.session_state.detection_info:
                boxes = st.session_state.detection_info.boxes
                st.metric("Objects Detected", len(boxes))
                
                # Show detected classes
                if len(boxes) > 0:
                    classes = boxes.cls.cpu().numpy()
                    class_names = [st.session_state.detection_info.names[int(c)] for c in classes]
                    st.write("**Detected Objects:**", ", ".join(set(class_names)))


def show_video_detection_page(model):
    """Display video detection interface."""
    st.header("🎥 Video Object Detection")
    st.markdown("Upload a video to detect objects frame by frame using YOLO.")
    
    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    if uploaded_file is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("🔍 Detect Objects in Video", use_container_width=True):
            with st.spinner("Processing video... This may take a while."):
                output_path = detect_objects_video(model, video_path, conf_threshold)
                
                st.success("Detection complete!")
                st.video(output_path)
                
                # Provide download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )





def main():
    """Main application function."""
    # Initialize authentication
    auth_ui = AuthUI()
    
    # Show authentication page if not logged in
    if not auth_ui.show_auth_page():
        return
    
    # User is logged in - show main application
    st.title("🎯 YOLO Object Detection System")
    st.markdown("**Powered by YOLOv11** - Real-time object detection for images and videos")
    
    # Show logout button
    auth_ui.show_logout_button()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load YOLO model. Please ensure yolo11n.pt is in the project directory.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Choose a feature:",
            ["📷 Image Detection", "🎥 Video Detection"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This application uses YOLOv11 for real-time object detection.
        
        **Features:**
        - Image object detection
        - Video object detection
        """)
    
    # Show selected page
    if page == "📷 Image Detection":
        show_image_detection_page(model)
    elif page == "🎥 Video Detection":
        show_video_detection_page(model)


if __name__ == "__main__":
    main()
