"""
YOLO Object Detection for Videos
Detects objects in video files using YOLO11n model
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def detect_objects_in_video(model, video_path, output_dir, conf_threshold=0.25, show=False):
    """
    Detect objects in a video file
    
    Args:
        model: YOLO model instance
        video_path: Path to input video
        output_dir: Directory to save output
        conf_threshold: Confidence threshold for detections
        show: Whether to display the result in real-time
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"detected_{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n{'='*60}")
    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"{'='*60}\n")
    
    frame_count = 0
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Display progress
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)", end='\r')
        
        # Show frame if requested
        if show:
            cv2.imshow('YOLO Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing interrupted by user")
                break
    
    # Release resources
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"\n\nProcessing complete!")
    print(f"Saved to: {output_path}")
    
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
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
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
