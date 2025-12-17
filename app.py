"""
YOLO Object Detection Application with Authentication
Main Streamlit application integrating login/signup with object detection features.

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Separate classes for model loading, detection, and results
- Liskov Substitution Principle (LSP): Common interface for all detectors
- Dependency Inversion Principle (DIP): Depend on abstractions, not concrete implementations
"""

import streamlit as st
from auth import AuthUI
import os
from PIL import Image
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict


# ============================================================================
# DEPENDENCY INVERSION PRINCIPLE (DIP) - Model Loader Abstraction
# ============================================================================

class IModelLoader(ABC):
    """
    Abstract interface for model loading (DIP)
    High-level modules depend on this abstraction, not concrete YOLO implementation
    """
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load a model from the given path"""
        pass
    
    @abstractmethod
    def is_model_available(self, model_path: str) -> bool:
        """Check if model file exists"""
        pass


class YOLOModelLoader(IModelLoader):
    """
    Concrete implementation of IModelLoader for YOLO models (DIP)
    Can be replaced with other implementations without affecting dependent code
    """
    
    def load_model(self, model_path: str):
        """Load YOLO model from path"""
        if not self.is_model_available(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        return YOLO(model_path)
    
    def is_model_available(self, model_path: str) -> bool:
        """Check if YOLO model file exists"""
        return os.path.exists(model_path)


# ============================================================================
# LISKOV SUBSTITUTION PRINCIPLE (LSP) - Detector Interface
# ============================================================================

class IDetector(ABC):
    """
    Abstract base class for all detectors (LSP)
    Any detector that inherits from this can be substituted for another
    """
    
    def __init__(self, model, conf_threshold: float = 0.25):
        self.model = model
        self.conf_threshold = conf_threshold
    
    @abstractmethod
    def detect(self, source: Any) -> Dict[str, Any]:
        """Perform object detection on the given source"""
        pass
    
    @abstractmethod
    def get_detector_type(self) -> str:
        """Get the type of detector"""
        pass


# ============================================================================
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Image Detector
# ============================================================================

class ImageDetector(IDetector):
    """
    Image-specific detector (SRP + LSP)
    Single Responsibility: Only handles image object detection
    LSP: Can be substituted wherever IDetector is expected
    """
    
    def detect(self, source: Any) -> Dict[str, Any]:
        """Detect objects in an image"""
        # Convert PIL Image to numpy array if needed
        if isinstance(source, Image.Image):
            source = np.array(source)
        
        # Run inference
        results = self.model(source, conf=self.conf_threshold)
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        return {
            'annotated_image': annotated_image,
            'results': results[0],
            'detector_type': self.get_detector_type()
        }
    
    def get_detector_type(self) -> str:
        return "image"


# ============================================================================
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Video Detector
# ============================================================================

class VideoDetector(IDetector):
    """
    Video-specific detector (SRP + LSP)
    Single Responsibility: Only handles video object detection
    LSP: Can be substituted wherever IDetector is expected
    """
    
    def __init__(self, model, conf_threshold: float = 0.25, progress_callback=None):
        super().__init__(model, conf_threshold)
        self.progress_callback = progress_callback
    
    def detect(self, source: Any) -> Dict[str, Any]:
        """Detect objects in a video"""
        cap = cv2.VideoCapture(source)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary output file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()
            
            # Write frame
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            if self.progress_callback:
                self.progress_callback(frame_count / total_frames)
        
        cap.release()
        out.release()
        
        return {
            'output_path': output_path,
            'total_frames': total_frames,
            'fps': fps,
            'resolution': (width, height),
            'detector_type': self.get_detector_type()
        }
    
    def get_detector_type(self) -> str:
        return "video"


# ============================================================================
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Result Handler
# ============================================================================

class ResultHandler:
    """
    Handles formatting and displaying detection results (SRP)
    Single Responsibility: Process and format detection results
    """
    
    @staticmethod
    def get_detection_summary(results) -> Dict[str, Any]:
        """Extract summary information from detection results"""
        boxes = results.boxes
        num_detections = len(boxes)
        
        detected_classes = []
        if num_detections > 0:
            classes = boxes.cls.cpu().numpy()
            detected_classes = [results.names[int(c)] for c in classes]
        
        return {
            'num_detections': num_detections,
            'detected_classes': detected_classes,
            'unique_classes': list(set(detected_classes))
        }
    
    @staticmethod
    def format_detection_details(results):
        """Format detailed information for each detection"""
        details = []
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results.names[class_id]
            details.append({
                'index': i + 1,
                'class_name': class_name,
                'confidence': confidence
            })
        return details


# Page configuration
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Application Functions (Using SOLID Principles)
# ============================================================================

def load_model(model_loader: IModelLoader = None, model_path: str = "yolo11n.pt"):
    """
    Load YOLO model using DIP abstraction
    Depends on IModelLoader interface, not concrete YOLO class
    """
    if model_loader is None:
        model_loader = YOLOModelLoader()
    
    try:
        if not model_loader.is_model_available(model_path):
            st.error(f"Model file '{model_path}' not found!")
            return None
        return model_loader.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Old detection functions removed - now using ImageDetector and VideoDetector classes (SRP + LSP)


def show_image_detection_page(model):
    """
    Display image detection interface (SRP - UI only)
    Uses ImageDetector and ResultHandler following SOLID principles
    """
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
                    # Use ImageDetector class (SRP + LSP)
                    detector = ImageDetector(model, conf_threshold)
                    result = detector.detect(image)
                    
                    # Store results in session state
                    st.session_state.detection_result = result['annotated_image']
                    st.session_state.detection_info = result['results']
    
    with col2:
        if 'detection_result' in st.session_state:
            st.subheader("Detection Results")
            st.image(st.session_state.detection_result, caption="Detected Objects", use_container_width=True)
            
            # Use ResultHandler to format results (SRP)
            if st.session_state.detection_info:
                summary = ResultHandler.get_detection_summary(st.session_state.detection_info)
                st.metric("Objects Detected", summary['num_detections'])
                
                # Show detected classes
                if summary['num_detections'] > 0:
                    st.write("**Detected Objects:**", ", ".join(summary['unique_classes']))


def show_video_detection_page(model):
    """
    Display video detection interface (SRP - UI only)
    Uses VideoDetector following SOLID principles
    """
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
                # Use VideoDetector class with progress callback (SRP + LSP)
                progress_bar = st.progress(0)
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                detector = VideoDetector(model, conf_threshold, progress_callback=update_progress)
                result = detector.detect(video_path)
                
                progress_bar.empty()
                output_path = result['output_path']
                
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
