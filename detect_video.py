"""
YOLO Object Detection for Videos
Detects objects in video files using YOLO11n model

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Separate classes for detection and result handling
- Liskov Substitution Principle (LSP): VideoDetector implements common interface
- Dependency Inversion Principle (DIP): Depend on model loader abstraction
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from abc import ABC, abstractmethod
from typing import Dict, Any


# ============================================================================
# DEPENDENCY INVERSION PRINCIPLE (DIP) - Model Loader
# ============================================================================

class IModelLoader(ABC):
    """Abstract interface for model loading (DIP)"""
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load a model from the given path"""
        pass


class YOLOModelLoader(IModelLoader):
    """Concrete YOLO model loader (DIP)"""
    
    def load_model(self, model_path: str):
        """Load YOLO model from path"""
        return YOLO(model_path)


# ============================================================================
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Video Detector
# ============================================================================

class VideoDetector:
    """
    Video detector following SRP
    Single Responsibility: Detect objects in videos
    """
    
    def __init__(self, model, conf_threshold: float = 0.25):
        self.model = model
        self.conf_threshold = conf_threshold
    
    def detect(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """Detect objects in a video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Write frame to output
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        return {
            'output_path': output_path,
            'total_frames': frame_count,
            'fps': fps,
            'resolution': (width, height)
        }


# ============================================================================
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Result Handler
# ============================================================================

class VideoResultHandler:
    """
    Handles video result formatting and display (SRP)
    Single Responsibility: Format and display video detection results
    """
    
    @staticmethod
    def print_video_info(video_path: str, result: Dict[str, Any]):
        """Print video information"""
        width, height = result['resolution']
        print(f"\n{'='*60}")
        print(f"Video: {Path(video_path).name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {result['fps']}")
        print(f"Total Frames: {result['total_frames']}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def print_progress(frame_count: int, total_frames: int, current_fps: float):
        """Print processing progress"""
        progress = (frame_count / total_frames) * 100
        print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%) - FPS: {current_fps:.1f}", end='\r')
    
    @staticmethod
    def print_completion(output_path: str):
        """Print completion message"""
        print(f"\n\nProcessing complete!")
        print(f"Saved to: {output_path}")


def detect_objects_in_video(model, video_path, output_dir, conf_threshold=0.25, show=False):
    """
    Detect objects in a video file using SOLID principles
    """
    # Prepare output path
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"detected_{video_name}.mp4")
    
    # Use VideoDetector (SRP)
    detector = VideoDetector(model, conf_threshold)
    
    # Print video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Use VideoResultHandler (SRP)
    result_info = {
        'resolution': (width, height),
        'fps': fps,
        'total_frames': total_frames
    }
    VideoResultHandler.print_video_info(video_path, result_info)
    
    # Perform detection
    result = detector.detect(video_path, output_path)
    
    # Print completion
    VideoResultHandler.print_completion(output_path)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection for Videos")
    parser.add_argument("--source", type=str, default="input/video1.mp4",
                       help="Path to video file")
    parser.add_argument("--output", type=str, default="output/",
                       help="Output directory for annotated video")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                       help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (0-1)")
    parser.add_argument("--show", action="store_true",
                       help="Display real-time detection (press 'q' to stop)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load YOLO model using abstraction (DIP)
    print(f"Loading YOLO model: {args.model}")
    model_loader = YOLOModelLoader()
    model = model_loader.load_model(args.model)
    print("Model loaded successfully!")
    
    # Check if source is a valid file
    source_path = Path(args.source)
    
    if not source_path.is_file():
        print(f"Error: {args.source} is not a valid file")
        return
    
    # Process video
    detect_objects_in_video(model, str(source_path), args.output, args.conf, args.show)
    
    print(f"\n{'='*60}")
    print("Video processing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
