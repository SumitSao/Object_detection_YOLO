"""
YOLO Object Detection for Images
Detects objects in single or multiple images using YOLO11n model
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def detect_objects_in_image(model, image_path, output_dir, conf_threshold=0.25, show=False):
    """
    Detect objects in a single image
    
    Args:
        model: YOLO model instance
        image_path: Path to input image
        output_dir: Directory to save output
        conf_threshold: Confidence threshold for detections
        show: Whether to display the result
    """
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Get the result for the image
    result = results[0]
    
    # Plot results on image
    annotated_img = result.plot()
    
    # Save the annotated image
    image_name = Path(image_path).name
    output_path = os.path.join(output_dir, f"detected_{image_name}")
    cv2.imwrite(output_path, annotated_img)
    
    # Print detection summary
    num_detections = len(result.boxes)
    print(f"\n{'='*60}")
    print(f"Image: {image_name}")
    print(f"Detections: {num_detections}")
    print(f"{'='*60}")
    
    # Print details of each detection
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"{i+1}. {class_name}: {confidence:.2%}")
    
    print(f"\nSaved to: {output_path}")
    
    # Display if requested
    if show:
        cv2.imshow(f"Detection - {image_name}", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
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
