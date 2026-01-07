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
from utils.pdf_generator import PDFReportGenerator

# Activity monitoring imports
try:
    from activity_monitor import (
        ActivityClassifier,
        ActivityVisualizer,
        AlertManager,
        ActivityLogger
    )
    ACTIVITY_MONITOR_AVAILABLE = True
except ImportError:
    ACTIVITY_MONITOR_AVAILABLE = False


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
        
        # Create temporary output file with H.264 codec for better browser compatibility
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        # Use H.264 codec (avc1) which is widely supported by browsers
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
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

# Custom CSS for equal video heights
st.markdown("""
<style>
/* Make video containers equal height */
[data-testid="stVideo"] {
    height: 500px;
}
[data-testid="stVideo"] video {
    height: 100%;
    object-fit: contain;
}

/* Make image containers equal height */
[data-testid="stImage"] {
    height: 500px;
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stImage"] img {
    max-height: 100%;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)


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
    
    # Initialize file uploader key in session state
    if 'img_uploader_key' not in st.session_state:
        st.session_state.img_uploader_key = 0
    
    # File uploader and clear button
    col_upload, col_clear = st.columns([4, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            key=f"image_uploader_{st.session_state.img_uploader_key}"
        )
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing to align with uploader
        if st.button("🗑️ Clear", use_container_width=True, key="clear_image"):
            # Clear session state
            if 'detection_result' in st.session_state:
                del st.session_state.detection_result
            if 'detection_info' in st.session_state:
                del st.session_state.detection_info
            if 'original_image' in st.session_state:
                del st.session_state.original_image
            # Change uploader key to clear the file
            st.session_state.img_uploader_key += 1
            st.rerun()
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if st.button("🔍 Detect Objects", use_container_width=True):
            with st.spinner("Detecting objects..."):
                # Use ImageDetector class (SRP + LSP)
                detector = ImageDetector(model, conf_threshold)
                result = detector.detect(image)
                
                # Store results in session state
                st.session_state.detection_result = result['annotated_image']
                st.session_state.detection_info = result['results']
                st.session_state.original_image = image
        
        # Display images side by side if detection is complete
        if 'detection_result' in st.session_state and hasattr(st.session_state, 'original_image'):
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Uploaded Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("✅ Detected Image")
                st.image(st.session_state.detection_result, use_container_width=True)
            
            # Show detection summary below images
            st.markdown("---")
            
            # Use ResultHandler to format results (SRP)
            if st.session_state.detection_info:
                summary = ResultHandler.get_detection_summary(st.session_state.detection_info)
                
                # Show metrics in columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("🎯 Objects Detected", summary['num_detections'])
                with metric_col2:
                    st.metric("📊 Unique Classes", len(summary['unique_classes']))
                with metric_col3:
                    st.metric("✅ Status", "Complete")
                
                # Show detected classes
                if summary['num_detections'] > 0:
                    st.markdown("**🔍 Detected Objects:**")
                    st.info(", ".join(summary['unique_classes']))
                
                # Download button for detected image
                st.markdown("---")
                # Convert numpy array to PIL Image for download
                from PIL import Image as PILImage
                import io
                detected_pil = PILImage.fromarray(st.session_state.detection_result)
                buf = io.BytesIO()
                detected_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    st.download_button(
                        label="📥 Download Detected Image",
                        data=byte_im,
                        file_name="detected_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_download2:
                    # Generate PDF report
                    pdf_generator = PDFReportGenerator()
                    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
                    
                    # Get username from session state
                    username = st.session_state.get('username', 'User')
                    
                    # Generate PDF
                    pdf_generator.generate_image_detection_report(
                        original_image=image,
                        detected_image=st.session_state.detection_result,
                        detection_results=summary,
                        output_path=pdf_path,
                        username=username
                    )
                    
                    # Read PDF for download
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name="detection_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            # Show only uploaded image before detection
            st.markdown("---")
            st.subheader("📤 Uploaded Image")
            st.image(image, use_container_width=True)


def show_video_detection_page(model):
    """
    Display video detection interface (SRP - UI only)
    Uses VideoDetector following SOLID principles
    """
    st.header("🎥 Video Object Detection")
    st.markdown("Upload a video to detect objects frame by frame using YOLO.")
    
    # Initialize file uploader key in session state
    if 'vid_uploader_key' not in st.session_state:
        st.session_state.vid_uploader_key = 0
    
    # File uploader and clear button
    col_upload, col_clear = st.columns([4, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a video...", 
            type=['mp4', 'avi', 'mov'],
            key=f"video_uploader_{st.session_state.vid_uploader_key}"
        )
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing to align with uploader
        if st.button("🗑️ Clear", use_container_width=True, key="clear_video"):
            # Clear session state
            if 'video_output_path' in st.session_state:
                del st.session_state.video_output_path
            if 'video_input_path' in st.session_state:
                del st.session_state.video_input_path
            # Change uploader key to clear the file
            st.session_state.vid_uploader_key += 1
            st.rerun()
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    if uploaded_file is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        if st.button("🔍 Detect Objects in Video", use_container_width=True):
            with st.spinner("Processing video... This may take a while."):
                # Use VideoDetector class with progress callback (SRP + LSP)
                progress_bar = st.progress(0)
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                detector = VideoDetector(model, conf_threshold, progress_callback=update_progress)
                result = detector.detect(video_path)
                
                progress_bar.empty()
                
                # Store result in session state
                st.session_state.video_output_path = result['output_path']
                st.session_state.video_input_path = video_path
                st.success("Detection complete!")
        
        # Display videos side by side if detection is complete
        if 'video_output_path' in st.session_state and hasattr(st.session_state, 'video_input_path'):
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Uploaded Video")
                st.video(st.session_state.video_input_path)
            
            with col2:
                st.subheader("✅ Detected Video")
                st.video(st.session_state.video_output_path)
            
            # Download button below both videos
            st.markdown("---")
            with open(st.session_state.video_output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name="detected_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        else:
            # Show only uploaded video before detection
            st.markdown("---")
            st.subheader("📤 Uploaded Video")
            st.video(video_path)


def show_park_monitoring_page(model):
    """
    Display park activity monitoring interface
    """
    st.header("🏞️ Park Activity Monitoring")
    st.markdown("Monitor authorized and unauthorized activities in park surveillance.")
    
    if not ACTIVITY_MONITOR_AVAILABLE:
        st.error("Activity monitoring module is not available. Please check installation.")
        return
    
    # Info about activity monitoring
    with st.expander("ℹ️ About Activity Monitoring"):
        st.markdown("""
        This feature classifies detected activities as:
        
        **✅ Authorized Activities (GREEN boxes)**:
        - Walking / Jogging
        - Cycling on designated paths
        - Pet walking (person + dog)
        - Playing / Sports
        - Sitting on benches
        
        **❌ Unauthorized Activities (RED boxes)**:
        - Vehicles in park (car, truck, bus)
        - Motorcycles
        - Skateboarding
        
        Violations are logged and can be exported as reports.
        """)
    
    # Tab selection
    tab1, tab2 = st.tabs(["🖼️ Image Monitoring", "🎥 Video Monitoring"])
    
    with tab1:
        st.subheader("Image Activity Monitoring")
        
        # Initialize file uploader key in session state
        if 'park_uploader_key' not in st.session_state:
            st.session_state.park_uploader_key = 0
        
        # File uploader and clear button
        col_upload, col_clear = st.columns([4, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload park image", 
                type=['jpg', 'jpeg', 'png'], 
                key=f"park_img_{st.session_state.park_uploader_key}"
            )
        with col_clear:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing to align with uploader
            if st.button("🗑️ Clear", use_container_width=True, key="clear_park"):
                # Clear session state
                if 'park_result' in st.session_state:
                    del st.session_state.park_result
                if 'park_classification' in st.session_state:
                    del st.session_state.park_classification
                if 'park_classifier' in st.session_state:
                    del st.session_state.park_classifier
                if 'park_original_image' in st.session_state:
                    del st.session_state.park_original_image
                # Change uploader key to clear the file
                st.session_state.park_uploader_key += 1
                st.rerun()
        
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="park_img_conf")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            if st.button("🔍 Monitor Activities", use_container_width=True):
                with st.spinner("Analyzing activities..."):
                    # Convert to numpy array
                    img_array = np.array(image)
                    
                    # Run YOLO detection
                    results = model(img_array, conf=conf_threshold)
                    
                    # Classify activities
                    classifier = ActivityClassifier()
                    classification_results = classifier.classify_detections(results[0])
                    
                    # Create visualization
                    visualizer = ActivityVisualizer()
                    annotated_img = visualizer.create_annotated_image(
                        img_array, classification_results, show_summary=True
                    )
                    
                    # Store in session state
                    st.session_state.park_result = annotated_img
                    st.session_state.park_classification = classification_results
                    st.session_state.park_classifier = classifier
                    st.session_state.park_original_image = image
            
            # Display images side by side if monitoring is complete
            if 'park_result' in st.session_state and hasattr(st.session_state, 'park_original_image'):
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📤 Uploaded Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("🏞️ Monitoring Results")
                    st.image(st.session_state.park_result, use_container_width=True)
                
                # Show statistics below images
                st.markdown("---")
                results = st.session_state.park_classification
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("✅ Authorized", results['authorized_count'], delta=None, delta_color="normal")
                with col_b:
                    st.metric("❌ Unauthorized", results['unauthorized_count'], delta=None, delta_color="inverse")
                with col_c:
                    st.metric("📊 Total Detected", results['authorized_count'] + results['unauthorized_count'])
                
                # Show violations
                if results['unauthorized_count'] > 0:
                    st.warning(f"⚠️ {results['unauthorized_count']} violation(s) detected!")
                    
                    with st.expander("View Violation Details"):
                        for v in results['violations']:
                            st.write(f"- **{v['rule'].name}**: {v['class_name']} ({v['rule'].alert_level.value} alert)")
                
                # Download button for monitoring result
                st.markdown("---")
                from PIL import Image as PILImage
                import io
                monitored_pil = PILImage.fromarray(st.session_state.park_result)
                buf = io.BytesIO()
                monitored_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    st.download_button(
                        label="📥 Download Monitoring Result",
                        data=byte_im,
                        file_name="park_monitoring_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_download2:
                    # Generate PDF report
                    pdf_generator = PDFReportGenerator()
                    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
                    
                    # Get username from session state
                    username = st.session_state.get('username', 'User')
                    
                    # Generate PDF
                    pdf_generator.generate_park_monitoring_report(
                        original_image=image,
                        monitored_image=st.session_state.park_result,
                        classification_results=st.session_state.park_classification,
                        output_path=pdf_path,
                        username=username
                    )
                    
                    # Read PDF for download
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name="park_monitoring_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                # Show only uploaded image before monitoring
                st.markdown("---")
                st.subheader("📤 Uploaded Image")
                st.image(image, use_container_width=True)
    
    with tab2:
        st.subheader("Video Activity Monitoring")
        st.info("🚧 Video monitoring is available via command line. Use: `python detect_video.py --source video.mp4 --monitor-activities`")
        
        st.markdown("""
        **Command Line Usage:**
        ```bash
        # Monitor activities in a video
        python detect_video.py --source input/park_video.mp4 --monitor-activities
        
        # This will generate:
        # - Annotated video with color-coded boxes
        # - Violation report (CSV and JSON)
        # - Activity log
        ```
        """)





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
            ["📷 Image Detection", "🎥 Video Detection", "🏞️ Park Monitoring"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This application uses YOLOv11 for real-time object detection.
        
        **Features:**
        - Image object detection
        - Video object detection
        - Park activity monitoring
        """)
    
    # Show selected page
    if page == "📷 Image Detection":
        show_image_detection_page(model)
    elif page == "🎥 Video Detection":
        show_video_detection_page(model)
    elif page == "🏞️ Park Monitoring":
        show_park_monitoring_page(model)


if __name__ == "__main__":
    main()
