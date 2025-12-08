"""
YOLO Inference Script
Script for object detection on images, videos, and directories
Supports both pretrained and custom-trained models
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import time
import numpy as np

class YOLOInference:
    """YOLO Inference class for object detection"""
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLO model for inference
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        print(f"📦 Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"✅ Model loaded successfully!")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   IoU threshold: {iou_threshold}")
        
    def predict_image(self, image_path, save_path=None, show=False):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save output image (optional)
            show: Display the result
            
        Returns:
            results: Detection results
        """
        print(f"\n Processing image: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False
        )
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Print detections
        detections = results[0].boxes
        print(f"   Found {len(detections)} objects:")
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.model.names[cls_id]
            print(f"   - {cls_name}: {conf:.2f}")
        
        # Save result
        if save_path:
            cv2.imwrite(str(save_path), annotated_img)
            print(f"   💾 Saved to: {save_path}")
        
        # Display result
        if show:
            cv2.imshow('YOLO Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def predict_video(self, video_path, save_path=None, show=False):
        """
        Run inference on a video file
        
        Args:
            video_path: Path to input video
            save_path: Path to save output video (optional)
            show: Display real-time results
            
        Returns:
            None
        """
        print(f"\n🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        
        # Process video
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Add frame info
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            if show:
                cv2.imshow('YOLO Video Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Stopped by user")
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {current_fps:.1f}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"   💾 Saved to: {save_path}")
        if show:
            cv2.destroyAllWindows()
        
        print(f"✅ Video processing complete!")
        print(f"   Processed {frame_count} frames in {elapsed_time:.2f}s")
        print(f"   Average FPS: {frame_count/elapsed_time:.2f}")
    

    def predict_directory(self, dir_path, output_dir=None, show=False):
        """
        Run inference on all images in a directory
        
        Args:
            dir_path: Path to directory containing images
            output_dir: Directory to save results (optional)
            show: Display each result
            
        Returns:
            None
        """
        dir_path = Path(dir_path)
        print(f"\n📁 Processing directory: {dir_path}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(dir_path.glob(f'*{ext}'))
            image_files.extend(dir_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No images found in {dir_path}")
            return
        
        print(f"   Found {len(image_files)} images")
        
        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
            
            save_path = None
            if output_dir:
                save_path = output_dir / f"detected_{image_path.name}"
            
            self.predict_image(image_path, save_path, show)
        
        print(f"\n✅ Directory processing complete!")
        print(f"   Processed {len(image_files)} images")
        if output_dir:
            print(f"   Results saved to: {output_dir}")


def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description='YOLO Inference Script')
    
    # Input source
    parser.add_argument('--source', type=str, default='data/train/images',
                       help='Input source: image path, video path, or directory')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (0-1)')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory or file path')
    parser.add_argument('--show', action='store_true',
                       help='Display results in real-time')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['auto', 'image', 'video', 'directory'],
                       default='auto', help='Inference mode (auto-detect by default)')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = YOLOInference(args.model, args.conf, args.iou)
    
    # Determine mode
    source = args.source
    mode = args.mode
    
    if mode == 'auto':
        # Auto-detect mode
        if Path(source).is_dir():
            mode = 'directory'
        elif Path(source).is_file():
            ext = Path(source).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                mode = 'video'
            else:
                mode = 'image'
        else:
            print(f"❌ Error: Invalid source: {source}")
            return
    
    print(f"\n🎯 Mode: {mode.upper()}")
    print("=" * 60)
    
    # Run inference based on mode
    if mode == 'image':
        save_path = Path(args.output) / f"detected_{Path(source).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        inference.predict_image(source, save_path, args.show)
    
    elif mode == 'video':
        save_path = Path(args.output) / f"detected_{Path(source).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        inference.predict_video(source, save_path, args.show)
    
    elif mode == 'directory':
        output_dir = Path(args.output)
        inference.predict_directory(source, output_dir, args.show)
    
    print("\n" + "=" * 60)
    print("✅ Inference complete!")


if __name__ == "__main__":
    main()
