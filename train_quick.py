"""
Quick Start Training Script for YOLO
Simple script to start training immediately with default settings
"""

from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolo11n.pt')  # nano model (fastest, smallest)
# model = YOLO('yolo11s.pt')  # small model (better accuracy)
# model = YOLO('yolo11m.pt')  # medium model (even better accuracy)

# Train the model
print("Starting YOLO training...")
print("Dataset: 7 classes - bike, bus, car, cycle, man, rickshaw, traffic light")
print("-" * 60)

results = model.train(
    data='data/data.yaml',  # Path to dataset configuration
    epochs=30,              # Number of training epochs
    imgsz=640,              # Image size
    batch=16,               # Batch size (reduce to 8 if out of memory)
    device='',              # Auto-detect GPU/CPU
    project='runs/train',   # Save results to runs/train
    name='quick_train',     # Experiment name
    plots=True,             # Save training plots
    save=True,              # Save checkpoints
    verbose=True,           # Verbose output
)

print("\n Training complete!")
print(f"Results saved to: runs/train/quick_train/")
print(f"Best model: runs/train/quick_train/weights/best.pt")
print(f"Last model: runs/train/quick_train/weights/last.pt")

# Validate the trained model
print("\n Validating model...")
metrics = model.val()

print("\n Validation Results:")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

print("\n To use your trained model for detection:")
print("   Update detect_image.py or detect_video.py:")
print("   model = YOLO('runs/train/quick_train/weights/best.pt')")
