"""
Activity Logger
Logs all detected activities for reporting and analysis.

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Only handles activity logging
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json


class ActivityLogger:
    """
    Logs all detected activities (SRP)
    Single Responsibility: Activity logging and statistics
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize activity logger
        
        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.activities: List[Dict[str, Any]] = []
    
    def log_activity(self, classification: Dict[str, Any], frame_number: int = 0):
        """
        Log a single activity
        
        Args:
            classification: Classification dictionary from ActivityClassifier
            frame_number: Frame number (for video processing)
        """
        rule = classification['rule']
        
        activity_log = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': frame_number,
            'activity_name': rule.name,
            'class_name': classification['class_name'],
            'confidence': classification['confidence'],
            'is_authorized': classification['is_authorized'],
            'alert_level': rule.alert_level.value,
            'bbox': classification['bbox'].tolist()
        }
        
        self.activities.append(activity_log)
    
    def log_all_activities(self, classification_results: Dict[str, Any], frame_number: int = 0):
        """
        Log all activities from classification results
        
        Args:
            classification_results: Results from ActivityClassifier
            frame_number: Frame number (for video processing)
        """
        for classification in classification_results['classifications']:
            self.log_activity(classification, frame_number)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged activities
        
        Returns:
            Dictionary with statistics
        """
        if not self.activities:
            return {
                'total_activities': 0,
                'authorized_count': 0,
                'unauthorized_count': 0,
                'activity_breakdown': {}
            }
        
        authorized_count = sum(1 for a in self.activities if a['is_authorized'])
        unauthorized_count = len(self.activities) - authorized_count
        
        # Activity breakdown
        activity_breakdown = {}
        for activity in self.activities:
            name = activity['activity_name']
            if name not in activity_breakdown:
                activity_breakdown[name] = {
                    'count': 0,
                    'authorized': activity['is_authorized']
                }
            activity_breakdown[name]['count'] += 1
        
        return {
            'total_activities': len(self.activities),
            'authorized_count': authorized_count,
            'unauthorized_count': unauthorized_count,
            'activity_breakdown': activity_breakdown
        }
    
    def save_log(self, filename: str = None) -> str:
        """
        Save activity log to JSON file
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"activity_log_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        log_data = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'activities': self.activities
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return str(filepath)
    
    def clear_log(self):
        """Clear all logged activities"""
        self.activities = []
