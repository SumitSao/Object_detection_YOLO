"""
Rules Engine for Activity Classification
Defines and evaluates rules to classify activities as authorized or unauthorized.

SOLID Principles Applied:
- Single Responsibility Principle (SRP): Separate classes for rules, engine, and classifier
- Open/Closed Principle (OCP): Easy to add new rules without modifying existing code
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ActivityRule:
    """
    Represents a single activity rule (SRP)
    Single Responsibility: Define activity classification criteria
    """
    name: str
    classes: List[str]  # YOLO class names that trigger this rule
    authorized: bool  # True = authorized (green), False = unauthorized (red)
    color: Tuple[int, int, int]  # BGR color for bounding box
    alert_level: AlertLevel
    description: str = ""
    requires_all_classes: bool = False  # True = all classes must be present (e.g., person + dog)
    
    def matches(self, detected_classes: List[str]) -> bool:
        """
        Check if detected classes match this rule
        
        Args:
            detected_classes: List of YOLO class names detected
            
        Returns:
            True if rule matches
        """
        if self.requires_all_classes:
            # All classes in rule must be present (e.g., person AND dog for pet walking)
            return all(cls in detected_classes for cls in self.classes)
        else:
            # Any class in rule triggers match (e.g., car OR truck OR bus)
            return any(cls in detected_classes for cls in self.classes)


class RulesEngine:
    """
    Evaluates detected objects against activity rules (SRP)
    Single Responsibility: Rule evaluation and matching
    """
    
    def __init__(self):
        self.authorized_rules = self._create_authorized_rules()
        self.unauthorized_rules = self._create_unauthorized_rules()
    
    def _create_authorized_rules(self) -> List[ActivityRule]:
        """Define authorized activity rules"""
        return [
            ActivityRule(
                name="Walking",
                classes=["person"],
                authorized=True,
                color=(0, 255, 0),  # Green
                alert_level=AlertLevel.NONE,
                description="Pedestrian movement"
            ),
            ActivityRule(
                name="Cycling",
                classes=["bicycle"],
                authorized=True,
                color=(0, 255, 0),
                alert_level=AlertLevel.NONE,
                description="Riding bicycle on designated paths"
            ),
            ActivityRule(
                name="Pet Walking",
                classes=["person", "dog"],
                authorized=True,
                color=(0, 255, 0),
                alert_level=AlertLevel.NONE,
                description="Walking with pets on leash",
                requires_all_classes=True
            ),
            ActivityRule(
                name="Playing",
                classes=["person", "sports ball", "frisbee"],
                authorized=True,
                color=(0, 255, 0),
                alert_level=AlertLevel.NONE,
                description="Sports and recreational activities"
            ),
            ActivityRule(
                name="Sitting",
                classes=["person", "bench"],
                authorized=True,
                color=(0, 255, 0),
                alert_level=AlertLevel.NONE,
                description="Resting on benches",
                requires_all_classes=True
            )
        ]
    
    def _create_unauthorized_rules(self) -> List[ActivityRule]:
        """Define unauthorized activity rules"""
        return [
            ActivityRule(
                name="Vehicle in Park",
                classes=["car", "truck", "bus"],
                authorized=False,
                color=(0, 0, 255),  # Red
                alert_level=AlertLevel.HIGH,
                description="Unauthorized vehicle detected"
            ),
            ActivityRule(
                name="Motorcycle",
                classes=["motorcycle"],
                authorized=False,
                color=(0, 0, 255),
                alert_level=AlertLevel.HIGH,
                description="Illegal riding - motorcycle not allowed"
            ),
            ActivityRule(
                name="Skateboard",
                classes=["skateboard"],
                authorized=False,
                color=(0, 0, 255),
                alert_level=AlertLevel.MEDIUM,
                description="Skateboarding not permitted"
            )
        ]
    
    def evaluate(self, detected_class: str, all_detected_classes: List[str]) -> Tuple[ActivityRule, bool]:
        """
        Evaluate a detected class against all rules
        
        Args:
            detected_class: Single YOLO class name
            all_detected_classes: All classes detected in the frame
            
        Returns:
            Tuple of (matching_rule, is_authorized)
        """
        # Check unauthorized rules first (higher priority)
        for rule in self.unauthorized_rules:
            if detected_class in rule.classes:
                if rule.requires_all_classes:
                    if rule.matches(all_detected_classes):
                        return rule, False
                else:
                    return rule, False
        
        # Check authorized rules
        for rule in self.authorized_rules:
            if detected_class in rule.classes:
                if rule.requires_all_classes:
                    if rule.matches(all_detected_classes):
                        return rule, True
                else:
                    return rule, True
        
        # Default: treat as authorized if no rule matches
        default_rule = ActivityRule(
            name="Unknown Activity",
            classes=[detected_class],
            authorized=True,
            color=(0, 255, 0),
            alert_level=AlertLevel.NONE,
            description="Unclassified activity"
        )
        return default_rule, True
    
    def get_all_rules(self) -> List[ActivityRule]:
        """Get all rules (authorized + unauthorized)"""
        return self.authorized_rules + self.unauthorized_rules


class ActivityClassifier:
    """
    Classifies detected objects into authorized/unauthorized activities (SRP)
    Single Responsibility: Activity classification and statistics
    """
    
    def __init__(self, rules_engine: RulesEngine = None):
        self.rules_engine = rules_engine or RulesEngine()
        self.authorized_count = 0
        self.unauthorized_count = 0
        self.violations = []
    
    def classify_detections(self, results) -> Dict[str, Any]:
        """
        Classify all detections in YOLO results
        
        Args:
            results: YOLO detection results object
            
        Returns:
            Dictionary with classification results
        """
        classifications = []
        all_detected_classes = []
        
        # Get all detected class names
        if len(results.boxes) > 0:
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                all_detected_classes.append(class_name)
        
        # Classify each detection
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # Evaluate against rules
            rule, is_authorized = self.rules_engine.evaluate(class_name, all_detected_classes)
            
            classification = {
                'index': i,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'rule': rule,
                'is_authorized': is_authorized
            }
            
            classifications.append(classification)
            
            # Update counts
            if is_authorized:
                self.authorized_count += 1
            else:
                self.unauthorized_count += 1
                self.violations.append(classification)
        
        return {
            'classifications': classifications,
            'authorized_count': self.authorized_count,
            'unauthorized_count': self.unauthorized_count,
            'violations': self.violations,
            'total_detections': len(classifications)
        }
    
    def get_summary(self) -> str:
        """Get a text summary of classifications"""
        summary = f"Park Activity Monitoring Results:\n"
        summary += f"✅ Authorized Activities: {self.authorized_count}\n"
        summary += f"❌ Unauthorized Activities: {self.unauthorized_count}\n"
        
        if self.violations:
            summary += f"\nViolations Detected:\n"
            for v in self.violations:
                summary += f"  - {v['rule'].name}: {v['class_name']} ({v['rule'].alert_level.value} alert)\n"
        
        return summary
    
    def reset(self):
        """Reset counters and violations"""
        self.authorized_count = 0
        self.unauthorized_count = 0
        self.violations = []
