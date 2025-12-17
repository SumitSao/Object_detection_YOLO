"""
YOLO Object Detection for Images
Detects objects in single or multiple images using YOLO11n model

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Separate classes for detection and result handling
- Liskov Substitution Principle (LSP): ImageDetector implements common interface
- Dependency Inversion Principle (DIP): Depend on model loader abstraction
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


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
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Image Detector
# ============================================================================

class ImageDetector:
    """
    Image detector following SRP
    Single Responsibility: Detect objects in images
    """
    
    def __init__(self, model, conf_threshold: float = 0.25):
        self.model = model
        self.conf_threshold = conf_threshold
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Detect objects in an image"""
        results = self.model(image_path, conf=self.conf_threshold)
        result = results[0]
        annotated_img = result.plot()
        
        return {
            'annotated_image': annotated_img,
            'results': result,
            'image_name': Path(image_path).name
        }


# ============================================================================
# SINGLE RESPONSIBILITY PRINCIPLE (SRP) - Result Handler
# ============================================================================

class ResultHandler:
    """
    Handles result formatting and display (SRP)
    Single Responsibility: Format and save detection results
    """
    
    @staticmethod
    def save_result(annotated_img, image_name: str, output_dir: str) -> str:
        """Save annotated image to output directory"""
        output_path = os.path.join(output_dir, f"detected_{image_name}")
        cv2.imwrite(output_path, annotated_img)
        return output_path
    
    @staticmethod
    def print_summary(result, image_name: str):
        """Print detection summary"""
        num_detections = len(result.boxes)
        print(f"\n{'='*60}")
        print(f"Image: {image_name}")
        print(f"Detections: {num_detections}")
        print(f"{'='*60}")
        
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            print(f"{i+1}. {class_name}: {confidence:.2%}")
    
    @staticmethod
    def display_result(annotated_img, image_name: str):
        """Display result in window"""
        cv2.imshow(f"Detection - {image_name}", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_objects_in_image(model, image_path, output_dir, conf_threshold=0.25, show=False):
    """
    Detect objects in a single image using SOLID principles
    """
    # Use ImageDetector (SRP)
    detector = ImageDetector(model, conf_threshold)
    detection_result = detector.detect(image_path)
    
    # Use ResultHandler (SRP)
    output_path = ResultHandler.save_result(
        detection_result['annotated_image'],
        detection_result['image_name'],
        output_dir
    )
    
    ResultHandler.print_summary(detection_result['results'], detection_result['image_name'])
    print(f"\nSaved to: {output_path}")
    
    if show:
        ResultHandler.display_result(detection_result['annotated_image'], detection_result['image_name'])
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection for Images")
    parser.add_argument("--source", type=str, default="input/", 
                       help="Path to image file or directory")
    parser.add_argument("--output", type=str, default="output/",
                       help="Output directory for annotated images")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                       help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (0-1)")
    parser.add_argument("--show", action="store_true",
                       help="Display results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load YOLO model using abstraction (DIP)
    print(f"Loading YOLO model: {args.model}")
    model_loader = YOLOModelLoader()
    model = model_loader.load_model(args.model)
    print("Model loaded successfully!\n")
    
    # Check if source is a file or directory
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Process single image
        detect_objects_in_image(model, str(source_path), args.output, args.conf, args.show)
    elif source_path.is_dir():
        # Process all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {args.source}")
            return
        
        print(f"Found {len(image_files)} images to process\n")
        
        for img_file in image_files:
            detect_objects_in_image(model, str(img_file), args.output, args.conf, args.show)
    else:
        print(f"Error: {args.source} is not a valid file or directory")
        return
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
