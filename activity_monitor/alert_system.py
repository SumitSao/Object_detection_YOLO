"""
Alert System for Unauthorized Activities
Generates and manages alerts for park violations.

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Separate classes for alerts and alert management
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class Alert:
    """
    Represents a single violation alert (SRP)
    Single Responsibility: Store alert information
    """
    timestamp: datetime
    activity_name: str
    class_name: str
    alert_level: str
    confidence: float
    description: str
    bbox: List[float] = field(default_factory=list)
    frame_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'activity_name': self.activity_name,
            'class_name': self.class_name,
            'alert_level': self.alert_level,
            'confidence': self.confidence,
            'description': self.description,
            'bbox': self.bbox,
            'frame_number': self.frame_number
        }


class AlertManager:
    """
    Manages alerts and violations (SRP)
    Single Responsibility: Alert collection, storage, and reporting
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize alert manager
        
        Args:
            output_dir: Directory to save alert logs
        """
        self.alerts: List[Alert] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def add_alert(self, classification: Dict[str, Any], frame_number: int = 0):
        """
        Add a new alert from a classification
        
        Args:
            classification: Classification dictionary from ActivityClassifier
            frame_number: Frame number (for video processing)
        """
        if not classification['is_authorized']:
            rule = classification['rule']
            
            alert = Alert(
                timestamp=datetime.now(),
                activity_name=rule.name,
                class_name=classification['class_name'],
                alert_level=rule.alert_level.value,
                confidence=classification['confidence'],
                description=rule.description,
                bbox=classification['bbox'].tolist(),
                frame_number=frame_number
            )
            
            self.alerts.append(alert)
    
    def get_alert_count(self) -> int:
        """Get total number of alerts"""
        return len(self.alerts)
    
    def get_alerts_by_level(self, level: str) -> List[Alert]:
        """Get alerts filtered by severity level"""
        return [alert for alert in self.alerts if alert.alert_level == level]
    
    def get_summary(self) -> str:
        """Get text summary of all alerts"""
        if not self.alerts:
            return "No violations detected."
        
        summary = f"\n{'='*60}\n"
        summary += f"VIOLATION REPORT\n"
        summary += f"{'='*60}\n"
        summary += f"Total Violations: {len(self.alerts)}\n\n"
        
        # Group by activity type
        activity_counts = {}
        for alert in self.alerts:
            activity_counts[alert.activity_name] = activity_counts.get(alert.activity_name, 0) + 1
        
        summary += "Violations by Type:\n"
        for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"  - {activity}: {count}\n"
        
        summary += f"\nDetailed Violations:\n"
        for i, alert in enumerate(self.alerts, 1):
            summary += f"\n{i}. {alert.activity_name}\n"
            summary += f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"   Alert Level: {alert.alert_level}\n"
            summary += f"   Confidence: {alert.confidence:.2f}\n"
            if alert.frame_number > 0:
                summary += f"   Frame: {alert.frame_number}\n"
        
        summary += f"\n{'='*60}\n"
        return summary
    
    def save_to_json(self, filename: str = None) -> str:
        """
        Save alerts to JSON file
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"violations_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        alerts_data = {
            'total_violations': len(self.alerts),
            'generated_at': datetime.now().isoformat(),
            'violations': [alert.to_dict() for alert in self.alerts]
        }
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        return str(filepath)
    
    def save_to_csv(self, filename: str = None) -> str:
        """
        Save alerts to CSV file
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"violations_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("Timestamp,Activity,Class,Alert Level,Confidence,Description,Frame\n")
            
            # Write data
            for alert in self.alerts:
                f.write(f"{alert.timestamp.isoformat()},")
                f.write(f"{alert.activity_name},")
                f.write(f"{alert.class_name},")
                f.write(f"{alert.alert_level},")
                f.write(f"{alert.confidence:.2f},")
                f.write(f"\"{alert.description}\",")
                f.write(f"{alert.frame_number}\n")
        
        return str(filepath)
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
