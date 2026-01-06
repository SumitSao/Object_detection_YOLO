"""
Activity Visualizer
Custom visualization with color-coded bounding boxes for authorized/unauthorized activities.

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Only handles visualization
"""

import cv2
import numpy as np
from typing import Dict, Any, List


class ActivityVisualizer:
    """
    Draws color-coded bounding boxes and labels (SRP)
    Single Responsibility: Visualization of activity classifications
    """
    
    def __init__(self, show_labels: bool = True, show_confidence: bool = True):
        """
        Initialize visualizer
        
        Args:
            show_labels: Whether to show activity labels
            show_confidence: Whether to show confidence scores
        """
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.label_thickness = 1
    
    def draw_classification(self, image: np.ndarray, classification: Dict[str, Any]) -> np.ndarray:
        """
        Draw a single classification on the image
        
        Args:
            image: Input image (BGR format)
            classification: Classification dictionary from ActivityClassifier
            
        Returns:
            Image with drawn bounding box and label
        """
        bbox = classification['bbox']
        rule = classification['rule']
        is_authorized = classification['is_authorized']
        confidence = classification['confidence']
        
        # Extract coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color from rule
        color = rule.color
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.thickness)
        
        # Prepare label
        if self.show_labels:
            status_symbol = "✓" if is_authorized else "✗"
            label = f"{rule.name} {status_symbol}"
            
            if self.show_confidence:
                label += f" {confidence:.2f}"
            
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.label_thickness
            )
            
            # Draw label background
            label_y = max(y1 - 10, label_height + 10)
            cv2.rectangle(
                image,
                (x1, label_y - label_height - baseline - 5),
                (x1 + label_width + 10, label_y + baseline),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1 + 5, label_y - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.label_thickness,
                cv2.LINE_AA
            )
        
        return image
    
    def draw_all_classifications(self, image: np.ndarray, classification_results: Dict[str, Any]) -> np.ndarray:
        """
        Draw all classifications on the image
        
        Args:
            image: Input image (BGR format)
            classification_results: Results from ActivityClassifier.classify_detections()
            
        Returns:
            Image with all bounding boxes and labels
        """
        # Create a copy to avoid modifying original
        annotated_image = image.copy()
        
        # Draw each classification
        for classification in classification_results['classifications']:
            annotated_image = self.draw_classification(annotated_image, classification)
        
        return annotated_image
    
    def add_summary_overlay(self, image: np.ndarray, classification_results: Dict[str, Any]) -> np.ndarray:
        """
        Add summary statistics overlay to the image
        
        Args:
            image: Input image
            classification_results: Results from ActivityClassifier
            
        Returns:
            Image with summary overlay
        """
        annotated_image = image.copy()
        
        # Prepare summary text
        authorized = classification_results['authorized_count']
        unauthorized = classification_results['unauthorized_count']
        total = classification_results['total_detections']
        
        summary_lines = [
            f"Total Detections: {total}",
            f"Authorized: {authorized}",
            f"Unauthorized: {unauthorized}"
        ]
        
        # Draw semi-transparent background
        overlay = annotated_image.copy()
        padding = 10
        line_height = 30
        bg_height = len(summary_lines) * line_height + 2 * padding
        bg_width = 300
        
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + bg_width, 10 + bg_height),
            (0, 0, 0),
            -1
        )
        
        # Blend overlay
        alpha = 0.6
        annotated_image = cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0)
        
        # Draw text
        y_offset = 10 + padding + 20
        for line in summary_lines:
            cv2.putText(
                annotated_image,
                line,
                (20, y_offset),
                self.font,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y_offset += line_height
        
        return annotated_image
    
    def create_annotated_image(
        self, 
        image: np.ndarray, 
        classification_results: Dict[str, Any],
        show_summary: bool = True
    ) -> np.ndarray:
        """
        Create fully annotated image with classifications and summary
        
        Args:
            image: Input image
            classification_results: Results from ActivityClassifier
            show_summary: Whether to show summary overlay
            
        Returns:
            Fully annotated image
        """
        # Draw all classifications
        annotated_image = self.draw_all_classifications(image, classification_results)
        
        # Add summary overlay if requested
        if show_summary:
            annotated_image = self.add_summary_overlay(annotated_image, classification_results)
        
        return annotated_image
