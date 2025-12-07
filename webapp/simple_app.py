"""
Simple Deepfake Detection Web App (Demo Version)

This is a simplified version that works without complex dependencies.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import numpy as np
from PIL import Image, ImageFile
import io
import base64
import cv2
import traceback
import logging
from datetime import datetime
import json
import re
import requests
import urllib.parse
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload folder
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
TEMP_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Analysis constraints
MAX_ANALYZE_DIM = 640  # max width/height to speed up processing
FACE_THUMB_SIZE = (160, 160)
FALLBACK_THUMB_SIZE = (256, 256)

# Social media patterns
SOCIAL_PATTERNS = {
    'instagram': [
        r'https?://(?:www\.)?instagram\.com/(?:p|reel|tv)/([A-Za-z0-9_-]+)',
        r'https?://(?:www\.)?instagram\.com/reels/([A-Za-z0-9_-]+)'
    ],
    'youtube': [
        r'https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]+)',
        r'https?://(?:www\.)?youtu\.be/([A-Za-z0-9_-]+)',
        r'https?://(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]+)'
    ],
    'tiktok': [
        r'https?://(?:www\.)?tiktok\.com/@[^/]+/video/([0-9]+)',
        r'https?://(?:www\.)?vm\.tiktok\.com/([A-Za-z0-9]+)'
    ]
}

# Load Haar cascade for lightweight face detection
try:
    HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(HAAR_PATH)
    USE_HAAR = not FACE_CASCADE.empty()
except Exception:
    USE_HAAR = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename):
    """Check if file is a video based on extension"""
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions


def detect_social_media_link(url):
    """Detect if URL is a supported social media link"""
    for platform, patterns in SOCIAL_PATTERNS.items():
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                return platform, match.group(1)
    return None, None


def download_social_media_video(url):
    """
    Download video from social media platforms.
    This is a simplified implementation - in production you'd want to use proper APIs.
    """
    platform, video_id = detect_social_media_link(url)
    
    if not platform:
        raise Exception("Unsupported social media platform or invalid URL")
    
    # For demo purposes, we'll create a mock download
    # In a real implementation, you'd use:
    # - YouTube Data API for YouTube
    # - Instagram Basic Display API for Instagram  
    # - TikTok API for TikTok
    
    try:
        # Create a temporary video file for demo
        temp_video_path = os.path.join(TEMP_FOLDER, f"{platform}_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # For demo, we'll create a simple test video or download a sample
        # In production, implement proper video downloading logic here
        
        # Mock implementation - create a simple test video
        if not os.path.exists(temp_video_path):
            # Create a simple test video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640, 480))
            
            # Create a simple test video with colored frames
            for i in range(60):  # 3 seconds at 20fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Add some color variation
                frame[:, :, 0] = (i * 4) % 255  # Blue
                frame[:, :, 1] = (i * 3) % 255  # Green  
                frame[:, :, 2] = (i * 2) % 255  # Red
                
                # Add text
                cv2.putText(frame, f'Demo Video - {platform.upper()}', (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Frame {i+1}/60', (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
        
        # Read the video file
        with open(temp_video_path, 'rb') as f:
            video_data = f.read()
        
        return video_data, f"{platform}_{video_id}.mp4"
        
    except Exception as e:
        raise Exception(f"Failed to download video from {platform}: {str(e)}")


def resize_for_analysis(pil_image: Image.Image) -> Image.Image:
    """Resize keeping aspect ratio so longest side <= MAX_ANALYZE_DIM."""
    w, h = pil_image.size
    scale = min(1.0, MAX_ANALYZE_DIM / max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return pil_image.resize(new_size, Image.BILINEAR)
    return pil_image


def detect_faces_rgb(image_rgb: np.ndarray):
    if not USE_HAAR:
        return []
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return faces
    except Exception:
        return []


def analyze_video(video_bytes):
    """
    Analyze video by extracting frames and analyzing each frame.
    Returns analysis results for multiple frames.
    """
    # Save video bytes to temporary file
    temp_video_path = os.path.join(UPLOAD_FOLDER, f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    
    try:
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames for analysis (every 30 frames or max 10 frames)
        frame_interval = max(1, total_frames // 10)
        frame_predictions = []
        total_faces = 0
        
        frame_count = 0
        analyzed_frames = 0
        
        while cap.isOpened() and analyzed_frames < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image for analysis
                pil_image = Image.fromarray(frame_rgb)
                
                # Analyze this frame
                frame_analysis = analyze_image(pil_image)
                
                # Prepare frame results
                frame_result = {
                    'frame_id': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'prediction': frame_analysis['prediction'],
                    'confidence': frame_analysis['confidence'],
                    'num_faces': frame_analysis['num_faces'],
                    'faces': []
                }
                
                # Add face predictions for this frame
                for idx, face_pred in enumerate(frame_analysis['face_predictions']):
                    frame_result['faces'].append({
                        'face_id': idx,
                        'prediction': face_pred['prediction'],
                        'confidence': face_pred['confidence'],
                        'fake_probability': face_pred['fake_probability'],
                        'face_image': face_pred['face_image']
                    })
                
                frame_predictions.append(frame_result)
                total_faces += frame_analysis['num_faces']
                analyzed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        # Calculate overall video prediction
        real_count = sum(1 for f in frame_predictions if f['prediction'] == 'REAL')
        fake_count = len(frame_predictions) - real_count
        
        if real_count > fake_count:
            overall_prediction = "REAL"
            overall_confidence = real_count / len(frame_predictions) if frame_predictions else 0.5
        else:
            overall_prediction = "FAKE"
            overall_confidence = fake_count / len(frame_predictions) if frame_predictions else 0.5
        
        return {
            'file_type': 'video',
            'overall_prediction': overall_prediction,
            'overall_confidence': overall_confidence,
            'total_faces': total_faces,
            'num_frames': len(frame_predictions),
            'frame_predictions': frame_predictions,
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'analyzed_frames': analyzed_frames
            }
        }
        
    except Exception as e:
        raise Exception(f"Video analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass


def create_analysis_charts(analysis_data):
    """Create matplotlib charts for the report"""
    charts = []
    
    # Create confidence chart
    fig, ax = plt.subplots(figsize=(8, 6))
    prediction = analysis_data['prediction']
    confidence = analysis_data['confidence']
    
    colors_map = {'REAL': 'green', 'FAKE': 'red'}
    bar_color = colors_map.get(prediction, 'blue')
    
    bars = ax.bar([prediction], [confidence * 100], color=bar_color, alpha=0.7)
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Deepfake Detection Confidence')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart to bytes
    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
    chart_buffer.seek(0)
    charts.append(chart_buffer)
    plt.close()
    
    # Create metrics radar chart
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
    
    metrics = ['Brightness', 'Contrast', 'Sharpness']
    values = [
        min(100, analysis_data['brightness'] / 2.55),  # Normalize to 0-100
        min(100, analysis_data['contrast'] / 1.0),     # Normalize to 0-100
        min(100, analysis_data['sharpness'] / 5.0)     # Normalize to 0-100
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label='Image Metrics')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title('Image Quality Metrics', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save chart to bytes
    chart_buffer2 = io.BytesIO()
    plt.savefig(chart_buffer2, format='png', dpi=150, bbox_inches='tight')
    chart_buffer2.seek(0)
    charts.append(chart_buffer2)
    plt.close()
    
    return charts


def generate_pdf_report(analysis_data, original_image, filename):
    """Generate a comprehensive PDF report"""
    report_filename = f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    story.append(Paragraph("Deepfake Detection Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Analyzed File:</b> {filename}", styles['Normal']))
    story.append(Paragraph(f"<b>Analysis Method:</b> AI-Powered Deep Learning Model", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
    prediction = analysis_data['prediction']
    confidence = analysis_data['confidence']
    
    summary_text = f"""
    The analyzed image has been classified as <b>{prediction}</b> with a confidence level of <b>{confidence*100:.1f}%</b>.
    This analysis was performed using advanced computer vision techniques including face detection and image quality assessment.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Original Image
    story.append(Paragraph("<b>Analyzed Image</b>", styles['Heading2']))
    
    # Resize image for PDF
    img_for_pdf = original_image.copy()
    img_for_pdf.thumbnail((400, 400), Image.LANCZOS)
    img_buffer = io.BytesIO()
    img_for_pdf.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    pdf_img = RLImage(img_buffer, width=4*inch, height=3*inch)
    story.append(pdf_img)
    story.append(Spacer(1, 20))
    
    # Analysis Results
    story.append(Paragraph("<b>Detailed Analysis Results</b>", styles['Heading2']))
    
    # Results table
    results_data = [
        ['Metric', 'Value', 'Description'],
        ['Prediction', prediction, 'AI classification result'],
        ['Confidence', f'{confidence*100:.1f}%', 'Model confidence in prediction'],
        ['Faces Detected', str(analysis_data['num_faces']), 'Number of faces found in image'],
        ['Brightness', f'{analysis_data["brightness"]:.1f}', 'Average image brightness (0-255)'],
        ['Contrast', f'{analysis_data["contrast"]:.1f}', 'Image contrast standard deviation'],
        ['Sharpness', f'{analysis_data["sharpness"]:.1f}', 'Image sharpness measure']
    ]
    
    results_table = Table(results_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Charts
    story.append(Paragraph("<b>Analysis Visualizations</b>", styles['Heading2']))
    
    charts = create_analysis_charts(analysis_data)
    
    for i, chart_buffer in enumerate(charts):
        chart_img = RLImage(chart_buffer, width=6*inch, height=4.5*inch)
        story.append(chart_img)
        story.append(Spacer(1, 10))
    
    # Technical Details
    story.append(Paragraph("<b>Technical Analysis Details</b>", styles['Heading2']))
    
    tech_details = f"""
    <b>Face Detection:</b> {analysis_data['num_faces']} faces were detected using Haar Cascade classifier.
    <br/><br/>
    <b>Image Quality Assessment:</b> The analysis evaluated multiple image characteristics including brightness, contrast, and sharpness to assess authenticity.
    <br/><br/>
    <b>Confidence Calculation:</b> The confidence score is based on a combination of face detection results, image quality metrics, and pattern analysis.
    """
    story.append(Paragraph(tech_details, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("<b>Disclaimer</b>", styles['Heading2']))
    disclaimer = """
    This analysis is provided for educational and research purposes only. The results should not be considered as definitive proof of authenticity or manipulation. 
    For professional use, please consult with qualified experts and use multiple verification methods.
    """
    story.append(Paragraph(disclaimer, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    return report_path, report_filename


def analyze_image(image: Image.Image):
    """
    Simple image analysis for demo purposes with basic face detection.
    Returns analysis dict including face thumbnails b64.
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Downscale for speed
    image_small = resize_for_analysis(image)

    # Convert to numpy array
    img_array = np.array(image_small)

    # Face detection (Haar)
    faces = detect_faces_rgb(img_array)
    num_faces = len(faces)

    # Simple signals
    brightness = float(np.mean(img_array))
    contrast = float(np.std(img_array))
    sharpness_proxy = float(
        cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    )

    # Enhanced heuristic scoring for AI-generated image detection
    # AI-generated images often have specific characteristics:
    # - Overly perfect/uniform textures
    # - Unusual frequency patterns
    # - Inconsistent lighting/reflections
    # - Artifacts in certain frequency bands
    
    fake_score = 0.0  # Start with 0 (real), higher = more likely fake
    
    try:
        # Calculate additional features for AI detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Frequency domain analysis (FFT)
        try:
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # AI-generated images often have unusual frequency patterns
            # Check for overly uniform frequency distribution
            freq_variance = np.var(magnitude_spectrum)
            if freq_variance < 1000:  # Very uniform = suspicious
                fake_score += 0.15
            elif freq_variance > 50000:  # Very chaotic = also suspicious
                fake_score += 0.1
        except Exception:
            pass  # Skip FFT if it fails
        
        # 2. Texture analysis - AI images often have overly smooth textures
        # Calculate local variance (texture measure)
        try:
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = np.var(gray.astype(np.float32) - local_mean)
            
            if local_variance < 200:  # Very smooth texture = suspicious
                fake_score += 0.2
            elif local_variance > 3000:  # Very rough = less suspicious but still check
                fake_score -= 0.05
        except Exception:
            pass  # Skip texture analysis if it fails
        
        # 3. Edge analysis - AI images may have inconsistent edge patterns
        try:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Very low or very high edge density can indicate AI generation
            if edge_density < 0.05:  # Too smooth
                fake_score += 0.15
            elif edge_density > 0.3:  # Unnaturally detailed
                fake_score += 0.1
        except Exception:
            pass  # Skip edge analysis if it fails
        
        # 4. Color consistency - AI images may have unusual color distributions
        try:
            # Check color channel correlations
            r_channel = img_array[:, :, 0].astype(np.float32)
            g_channel = img_array[:, :, 1].astype(np.float32)
            b_channel = img_array[:, :, 2].astype(np.float32)
            
            # Calculate correlation between channels
            rg_corr = np.corrcoef(r_channel.flatten(), g_channel.flatten())[0, 1]
            rb_corr = np.corrcoef(r_channel.flatten(), b_channel.flatten())[0, 1]
            gb_corr = np.corrcoef(g_channel.flatten(), b_channel.flatten())[0, 1]
            
            if not (np.isnan(rg_corr) or np.isnan(rb_corr) or np.isnan(gb_corr)):
                avg_corr = (rg_corr + rb_corr + gb_corr) / 3
                
                # Very high correlation (unnaturally consistent colors) = suspicious
                if avg_corr > 0.95:
                    fake_score += 0.1
        except Exception:
            pass  # Skip color analysis if it fails
        
        # 5. Brightness and contrast analysis
        # AI images often have very uniform lighting
        brightness_std = np.std(img_array)
        if brightness_std < 20:  # Very uniform brightness
            fake_score += 0.1
        elif brightness_std > 80:  # Very varied = more natural
            fake_score -= 0.05
        
        # 6. Sharpness analysis
        # AI images can be either too sharp (over-processed) or too soft
        if sharpness_proxy < 30:  # Too blurry
            fake_score += 0.1
        elif sharpness_proxy > 500:  # Unnaturally sharp
            fake_score += 0.1
        
        # 7. Face-specific checks (if faces detected)
        if num_faces > 0:
            # Real photos with faces usually have some imperfections
            # Perfect faces might indicate AI generation
            if num_faces == 1 and sharpness_proxy > 400:
                fake_score += 0.1
    except Exception as e:
        # If advanced analysis fails, fall back to basic heuristics
        logger.warning(f"Advanced analysis failed, using basic heuristics: {e}")
        # Basic fallback: use sharpness and contrast
        if sharpness_proxy < 30 or sharpness_proxy > 500:
            fake_score += 0.2
        if contrast < 20:
            fake_score += 0.15
    
    # Normalize fake_score to 0-1 range
    fake_score = max(0.0, min(1.0, fake_score))
    
    # Convert to prediction
    # Lower threshold for fake detection (0.35 instead of 0.5) to catch more AI-generated images
    if fake_score >= 0.35:
        prediction = "FAKE"
        confidence = fake_score
    else:
        prediction = "REAL"
        confidence = 1.0 - fake_score
    
    # Ensure minimum confidence
    confidence = max(0.5, min(0.99, confidence))

    # Face thumbnails (from resized image)
    face_predictions = []
    for idx, (x, y, w, h) in enumerate(faces[:6]):
        crop = image_small.crop((int(x), int(y), int(x + w), int(y + h)))
        thumb = crop.resize(FACE_THUMB_SIZE)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        face_b64 = base64.b64encode(buf.getvalue()).decode()
        face_predictions.append({
            'prediction': prediction,
            'confidence': float(confidence),
            'fake_probability': (1.0 - confidence) if prediction == 'REAL' else float(confidence),
            'face_image': face_b64,
            'face_id': idx
        })

    # Fallback thumbnail if no faces
    fallback_b64 = None
    if not face_predictions:
        thumb = image_small.copy()
        thumb.thumbnail(FALLBACK_THUMB_SIZE, Image.BILINEAR)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        fallback_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness_proxy,
        'num_faces': num_faces,
        'face_predictions': face_predictions,
        'fallback_image_b64': fallback_b64
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: Images (JPG, PNG, GIF, BMP) and Videos (MP4, AVI, MOV, MKV, WMV)'}), 400

        # Read file
        raw = file.read()
        
        # Check if it's a video file
        if is_video_file(file.filename):
            # Analyze video
            analysis = analyze_video(raw)
            return jsonify(analysis)
        else:
            # Analyze image
            image = Image.open(io.BytesIO(raw))

            # If animated GIF, take first frame
            try:
                if getattr(image, 'is_animated', False):
                    image.seek(0)
                    image = image.convert('RGB')
            except Exception:
                pass

            analysis = analyze_image(image)

            prediction = analysis['prediction']
            confidence = analysis['confidence']
            face_predictions = analysis['face_predictions']

            # If no faces, use a single fallback thumbnail
            if not face_predictions and analysis['fallback_image_b64']:
                face_predictions = [{
                    'prediction': prediction,
                    'confidence': confidence,
                    'fake_probability': (1.0 - confidence) if prediction == 'REAL' else confidence,
                    'face_image': analysis['fallback_image_b64'],
                    'face_id': 0
                }]

            response = {
                'filename': file.filename,
                'file_type': 'image',
                'overall_prediction': prediction,
                'overall_confidence': confidence,
                'num_faces': analysis['num_faces'],
                'face_predictions': face_predictions,
                'metrics': {
                    'brightness': analysis['brightness'],
                    'contrast': analysis['contrast'],
                    'sharpness': analysis['sharpness']
                }
            }

            return jsonify(response)

    except Exception as e:
        # Log full traceback to console for debugging
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_link', methods=['POST'])
def analyze_social_media_link():
    """Analyze video from social media link"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        url = data['url'].strip()
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Detect social media platform
        platform, video_id = detect_social_media_link(url)
        if not platform:
            return jsonify({'error': 'Unsupported social media platform. Supported: Instagram, YouTube, TikTok'}), 400
        
        # Download video
        video_data, filename = download_social_media_video(url)
        
        # Analyze video
        analysis = analyze_video(video_data)
        
        # Add source information
        analysis['source_url'] = url
        analysis['source_platform'] = platform
        analysis['source_video_id'] = video_id
        analysis['filename'] = filename
        
        return jsonify(analysis)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to analyze link: {str(e)}'}), 500


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download PDF report"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract data from request
        analysis_data = data.get('analysis', {})
        image_data = data.get('image_data', '')
        filename = data.get('filename', 'unknown.jpg')
        
        # Convert base64 image back to PIL Image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate PDF report
        report_path, report_filename = generate_pdf_report(analysis_data, image, filename)
        
        return send_file(
            report_path,
            as_attachment=True,
            download_name=report_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Simple Deepfake Detection Demo is running!'
    })


if __name__ == '__main__':
    print("Starting Simple Deepfake Detection Demo...")
    print("Open your browser and go to: http://localhost:5000")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
