
# YOLO Object Detection System

A complete object detection system using YOLO11n for processing images and video clips with visualization and output saving capabilities.

## Features

- 🖼️ **Image Detection**: Process single images or batch process entire directories
- 🎥 **Video Detection**: Process video files with frame-by-frame object detection
- 📊 **Detailed Output**: Bounding boxes with labels and confidence scores
- 💾 **Save Results**: Automatically save annotated images and videos
- 🎯 **Configurable**: Adjust confidence thresholds and other parameters

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download YOLO Model**:
   The YOLO11n model (`yolo11n.pt`) will be automatically downloaded on first run.

## Project Structure

```
Object_detection_YOLO/
├── detect_image.py       # Image detection script
├── detect_video.py       # Video detection script
├── train_model.py        # Advanced model training script
├── train_quick.py        # Quick-start training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Training dataset
│   ├── data.yaml        # Dataset configuration
│   ├── train/           # Training images & labels
│   ├── valid/           # Validation images & labels
│   └── test/            # Test images & labels
├── input/               # Place your images/videos here
└── output/              # Detected results will be saved here
```

## Usage

### Model Training (Custom Dataset)

Train YOLO on your custom traffic detection dataset with 7 classes: bike, bus, car, cycle, man, rickshaw, traffic light.

**Quick Start Training** (Recommended for beginners):
```bash
python train_quick.py
```
- Trains for 50 epochs
- Auto-detects GPU/CPU
- Results saved to `runs/train/quick_train/`

**Advanced Training** (Full control):
```bash
python train_model.py
```
- 100 epochs with early stopping
- Configurable hyperparameters
- Data augmentation
- Checkpoint saving

**After Training**: Use your trained model by updating the model path in detection scripts:
```python
model = YOLO('runs/train/quick_train/weights/best.pt')
```

For detailed training guide, see the training documentation.


### Unified Inference (Recommended)

The `inference.py` script provides a unified interface for all detection tasks with advanced features.

**Single Image**:
```bash
python inference.py --source input/image.jpg --show
```

**Video**:
```bash
python inference.py --source input/video.mp4 --show
```

**Real-time Webcam** (press 'q' to quit, 's' for screenshot):
```bash
python inference.py --source 0 --show
```

**Batch Directory**:
```bash
python inference.py --source input/ --output results/
```

**With Custom Trained Model**:
```bash
python inference.py --source input/image.jpg --model runs/train/quick_train/weights/best.pt --show
```

**Advanced Options**:
```bash
python inference.py --source input/video.mp4 --conf 0.5 --iou 0.45 --show
```

### Image Detection (Legacy)

**Process a single image**:
```bash
python detect_image.py --source input/image.jpg
```

**Process all images in a directory**:
```bash
python detect_image.py --source input/
```

**With custom confidence threshold**:
```bash
python detect_image.py --source input/image.jpg --conf 0.5
```

**Display results**:
```bash
python detect_image.py --source input/image.jpg --show
```

### Video Detection

**Process a video**:
```bash
python detect_video.py --source input/video.mp4
```

**With real-time display** (press 'q' to stop):
```bash
python detect_video.py --source input/video.mp4 --show
```

**With custom confidence threshold**:
```bash
python detect_video.py --source input/video.mp4 --conf 0.4
```

## Command-Line Arguments

### Image Detection (`detect_image.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `input/` | Path to image file or directory |
| `--output` | `output/` | Output directory for results |
| `--model` | `yolo11n.pt` | YOLO model path |
| `--conf` | `0.25` | Confidence threshold (0-1) |
| `--show` | `False` | Display results |

### Video Detection (`detect_video.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `input/sample_video.mp4` | Path to video file |
| `--output` | `output/` | Output directory for results |
| `--model` | `yolo11n.pt` | YOLO model path |
| `--conf` | `0.25` | Confidence threshold (0-1) |
| `--show` | `False` | Display real-time detection |

## Output

- **Images**: Saved as `detected_<original_name>.jpg` in the output directory
- **Videos**: Saved as `detected_<original_name>.mp4` in the output directory
- **Console**: Displays detection summary with object classes and confidence scores

## Supported Formats

### Images
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

### Videos
- MP4
- AVI
- MOV
- MKV
- And other formats supported by OpenCV

## YOLO11n Model

This system uses YOLO11n (nano), which can detect 80 different object classes including:
- People
- Vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, bird, horse, etc.)
- Common objects (chair, table, laptop, phone, etc.)
- And many more!

For the complete list of detectable objects, see the [COCO dataset classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## Examples

### Example 1: Quick Start
```bash
# Create input directory and add your images/videos
mkdir input output

# Run detection on an image
python detect_image.py --source input/my_photo.jpg

# Check the output directory for results
```

### Example 2: Batch Processing
```bash
# Process all images in a folder
python detect_image.py --source input/ --conf 0.3
```

### Example 3: Video Processing
```bash
# Process a video with real-time preview
python detect_video.py --source input/my_video.mp4 --show
```

## Tips

- **Confidence Threshold**: Lower values (0.1-0.3) detect more objects but may include false positives. Higher values (0.5-0.7) are more conservative.
- **Performance**: YOLO11n is optimized for speed. For better accuracy, consider using larger models like `yolo11s.pt` or `yolo11m.pt`.
- **GPU Acceleration**: If you have a CUDA-compatible GPU, the model will automatically use it for faster processing.

## Troubleshooting

**Issue**: Model download fails
- **Solution**: Check your internet connection or manually download from [Ultralytics](https://github.com/ultralytics/assets/releases)

**Issue**: Video processing is slow
- **Solution**: Use a GPU or reduce video resolution

**Issue**: No objects detected
- **Solution**: Lower the confidence threshold using `--conf 0.1`

## License

This project uses the Ultralytics YOLO implementation, which is licensed under AGPL-3.0.
