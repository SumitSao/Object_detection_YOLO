"""
PDF Report Generator for YOLO Object Detection System

Generates comprehensive PDF reports with detection details, images,
and authorized/unauthorized activity classifications.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
from PIL import Image
import io
import numpy as np
from typing import Dict, Any, List, Optional


class PDFReportGenerator:
    """
    Generate professional PDF reports for object detection results.
    Supports both regular detection and park activity monitoring reports.
    """
    
    def __init__(self, page_size=letter):
        """
        Initialize PDF generator.
        
        Args:
            page_size: Page size for PDF (default: letter)
        """
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
    
    def _numpy_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if isinstance(img_array, np.ndarray):
            # Convert BGR to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1]  # BGR to RGB
            return Image.fromarray(img_array.astype('uint8'))
        return img_array
    
    def _pil_to_reportlab_image(self, pil_image: Image.Image, max_width: float = 3.5*inch, max_height: float = 3*inch) -> RLImage:
        """
        Convert PIL Image to ReportLab Image with size constraints.
        
        Args:
            pil_image: PIL Image object
            max_width: Maximum width in points
            max_height: Maximum height in points
            
        Returns:
            ReportLab Image object
        """
        # Save PIL image to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create ReportLab image
        rl_img = RLImage(img_buffer)
        
        # Calculate aspect ratio and resize
        aspect = pil_image.width / pil_image.height
        
        if pil_image.width > pil_image.height:
            rl_img.drawWidth = min(max_width, pil_image.width)
            rl_img.drawHeight = rl_img.drawWidth / aspect
            if rl_img.drawHeight > max_height:
                rl_img.drawHeight = max_height
                rl_img.drawWidth = rl_img.drawHeight * aspect
        else:
            rl_img.drawHeight = min(max_height, pil_image.height)
            rl_img.drawWidth = rl_img.drawHeight * aspect
            if rl_img.drawWidth > max_width:
                rl_img.drawWidth = max_width
                rl_img.drawHeight = rl_img.drawWidth / aspect
        
        return rl_img
    
    def _create_header_footer(self, canvas_obj, doc):
        """Add header and footer to each page."""
        canvas_obj.saveState()
        
        # Footer
        canvas_obj.setFont('Helvetica', 9)
        canvas_obj.setFillColor(colors.grey)
        canvas_obj.drawString(inch, 0.5 * inch, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas_obj.drawRightString(self.page_size[0] - inch, 0.5 * inch, f"Page {doc.page}")
        
        canvas_obj.restoreState()
    
    def generate_image_detection_report(
        self,
        original_image: Any,
        detected_image: Any,
        detection_results: Dict[str, Any],
        output_path: str,
        username: str = "User"
    ) -> str:
        """
        Generate PDF report for image detection.
        
        Args:
            original_image: Original image (PIL Image or numpy array)
            detected_image: Detected image with annotations (PIL Image or numpy array)
            detection_results: Dictionary containing detection summary
            output_path: Path to save the PDF
            username: Username of the person running detection
            
        Returns:
            Path to generated PDF
        """
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
        story = []
        
        # Title
        title = Paragraph("YOLO Object Detection Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata
        metadata_data = [
            ['Report Type:', 'Image Object Detection'],
            ['Generated By:', username],
            ['Date & Time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Detection Model:', 'YOLOv11']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f2ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detection Summary
        story.append(Paragraph("Detection Summary", self.styles['CustomSubtitle']))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Objects Detected', str(detection_results.get('num_detections', 0))],
            ['Unique Classes', str(len(detection_results.get('unique_classes', [])))],
            ['Detected Classes', ', '.join(detection_results.get('unique_classes', ['None']))]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Images Section
        story.append(Paragraph("Detection Results", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Convert images to PIL if needed
        original_pil = self._numpy_to_pil(original_image) if isinstance(original_image, np.ndarray) else original_image
        detected_pil = self._numpy_to_pil(detected_image) if isinstance(detected_image, np.ndarray) else detected_image
        
        # Create image table (side by side)
        original_rl = self._pil_to_reportlab_image(original_pil)
        detected_rl = self._pil_to_reportlab_image(detected_pil)
        
        image_data = [
            [Paragraph("<b>Original Image</b>", self.styles['Normal']), 
             Paragraph("<b>Detected Image</b>", self.styles['Normal'])],
            [original_rl, detected_rl]
        ]
        
        image_table = Table(image_data, colWidths=[3.5*inch, 3.5*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0'))
        ]))
        story.append(image_table)
        
        # Build PDF
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)
        
        return output_path
    
    def generate_park_monitoring_report(
        self,
        original_image: Any,
        monitored_image: Any,
        classification_results: Dict[str, Any],
        output_path: str,
        username: str = "User"
    ) -> str:
        """
        Generate PDF report for park activity monitoring.
        
        Args:
            original_image: Original image (PIL Image or numpy array)
            monitored_image: Monitored image with color-coded annotations
            classification_results: Dictionary containing classification results
            output_path: Path to save the PDF
            username: Username of the person running monitoring
            
        Returns:
            Path to generated PDF
        """
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
        story = []
        
        # Title
        title = Paragraph("Park Activity Monitoring Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata
        metadata_data = [
            ['Report Type:', 'Park Activity Monitoring'],
            ['Generated By:', username],
            ['Date & Time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Detection Model:', 'YOLOv11 + Activity Classifier']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f2ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Activity Summary
        story.append(Paragraph("Activity Summary", self.styles['CustomSubtitle']))
        
        authorized_count = classification_results.get('authorized_count', 0)
        unauthorized_count = classification_results.get('unauthorized_count', 0)
        total_count = authorized_count + unauthorized_count
        
        summary_data = [
            ['Metric', 'Count', 'Status'],
            ['Authorized Activities', str(authorized_count), '✓ Allowed'],
            ['Unauthorized Activities', str(unauthorized_count), '✗ Violation'],
            ['Total Detections', str(total_count), '-']
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d4edda')),  # Green for authorized
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8d7da')),  # Red for unauthorized
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Authorized Activities Details
        if classification_results.get('authorized', []):
            story.append(Paragraph("Authorized Activities (Green Boxes)", self.styles['SectionHeader']))
            
            auth_data = [['Class Name', 'Confidence', 'Activity Type']]
            for activity in classification_results['authorized']:
                auth_data.append([
                    activity['class_name'],
                    f"{activity['confidence']:.2%}",
                    'Allowed'
                ])
            
            auth_table = Table(auth_data, colWidths=[2*inch, 2*inch, 2*inch])
            auth_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')])
            ]))
            story.append(auth_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Unauthorized Activities / Violations
        if classification_results.get('violations', []):
            story.append(Paragraph("⚠️ Violations Detected (Red Boxes)", self.styles['SectionHeader']))
            
            viol_data = [['Class Name', 'Confidence', 'Rule Violated', 'Alert Level']]
            for violation in classification_results['violations']:
                viol_data.append([
                    violation['class_name'],
                    f"{violation['confidence']:.2%}",
                    violation['rule'].name,
                    violation['rule'].alert_level.value.upper()
                ])
            
            viol_table = Table(viol_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            viol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8d7da')])
            ]))
            story.append(viol_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Images Section
        story.append(Paragraph("Monitoring Results", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Convert images to PIL if needed
        original_pil = self._numpy_to_pil(original_image) if isinstance(original_image, np.ndarray) else original_image
        monitored_pil = self._numpy_to_pil(monitored_image) if isinstance(monitored_image, np.ndarray) else monitored_image
        
        # Create image table (side by side)
        original_rl = self._pil_to_reportlab_image(original_pil)
        monitored_rl = self._pil_to_reportlab_image(monitored_pil)
        
        image_data = [
            [Paragraph("<b>Original Image</b>", self.styles['Normal']), 
             Paragraph("<b>Monitored Image</b><br/>(Green=Authorized, Red=Unauthorized)", self.styles['Normal'])],
            [original_rl, monitored_rl]
        ]
        
        image_table = Table(image_data, colWidths=[3.5*inch, 3.5*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0'))
        ]))
        story.append(image_table)
        
        # Build PDF
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)
        
        return output_path
    
    def generate_park_video_monitoring_report(
        self,
        classification_results: Dict[str, Any],
        output_path: str,
        username: str = "User",
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate PDF report for park video activity monitoring.
        
        Args:
            classification_results: Dictionary containing classification results
            output_path: Path to save the PDF
            username: Username of the person running monitoring
            video_metadata: Optional video metadata (fps, total_frames, duration)
            
        Returns:
            Path to generated PDF
        """
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=self.page_size)
        story = []
        
        # Title
        title = Paragraph("Park Video Activity Monitoring Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata
        metadata_data = [
            ['Report Type:', 'Park Video Activity Monitoring'],
            ['Generated By:', username],
            ['Date & Time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Detection Model:', 'YOLOv11 + Activity Classifier']
        ]
        
        # Add video metadata if available
        if video_metadata:
            if video_metadata.get('total_frames'):
                metadata_data.append(['Total Frames:', str(video_metadata['total_frames'])])
            if video_metadata.get('fps'):
                metadata_data.append(['FPS:', str(video_metadata['fps'])])
            if video_metadata.get('duration'):
                duration_sec = video_metadata['duration']
                metadata_data.append(['Duration:', f"{duration_sec:.1f} seconds"])
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f2ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Activity Summary
        story.append(Paragraph("Activity Summary", self.styles['CustomSubtitle']))
        
        authorized_count = classification_results.get('authorized_count', 0)
        unauthorized_count = classification_results.get('unauthorized_count', 0)
        total_count = authorized_count + unauthorized_count
        
        summary_data = [
            ['Metric', 'Count', 'Status'],
            ['Authorized Activities', str(authorized_count), '✓ Allowed'],
            ['Unauthorized Activities', str(unauthorized_count), '✗ Violation'],
            ['Total Detections', str(total_count), '-']
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d4edda')),  # Green for authorized
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8d7da')),  # Red for unauthorized
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Violations Details
        if classification_results.get('violations', []):
            story.append(Paragraph("⚠️ Violations Detected", self.styles['SectionHeader']))
            
            viol_data = [['Class Name', 'Confidence', 'Rule Violated', 'Alert Level']]
            for violation in classification_results['violations']:
                viol_data.append([
                    violation['class_name'],
                    f"{violation['confidence']:.2%}",
                    violation['rule'].name,
                    violation['rule'].alert_level.value.upper()
                ])
            
            viol_table = Table(viol_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            viol_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8d7da')])
            ]))
            story.append(viol_table)
            story.append(Spacer(1, 0.3*inch))
        else:
            story.append(Paragraph("✅ No Violations Detected", self.styles['SectionHeader']))
            story.append(Paragraph("All detected activities were authorized.", self.styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
        
        # Notes section
        story.append(Paragraph("Notes", self.styles['CustomSubtitle']))
        notes_text = """
        <b>Color Coding:</b><br/>
        • <font color="green">Green boxes</font> indicate authorized activities (walking, cycling, pets, etc.)<br/>
        • <font color="red">Red boxes</font> indicate unauthorized activities (vehicles, motorcycles, etc.)<br/>
        <br/>
        <b>Recommendations:</b><br/>
        • Review violation frames for potential security concerns<br/>
        • Consider additional signage in high-violation areas<br/>
        • Monitor trends over time to identify patterns
        """
        story.append(Paragraph(notes_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)
        
        return output_path
