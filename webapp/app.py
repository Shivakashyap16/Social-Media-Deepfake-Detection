"""
Flask Web Application for Deepfake Detection

This module provides a web interface for uploading and analyzing
videos and images for deepfake detection.

Author: Deepfake Detection Team
Date: 2024
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
import cv2
import base64
import io

from src.model import create_model
from src.data_preprocessing import DataPreprocessor
from utils.metrics import calculate_metrics
from utils.logger import setup_logger

# Setup logging
logger = setup_logger('webapp')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'wmv'}
MODEL_PATH = '../models/best_model.pth'
CONFIG_PATH = '../config/model_config.yaml'

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
preprocessor = None
device = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model configuration
        if os.path.exists(CONFIG_PATH):
            import yaml
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'type': 'efficientnet',
                'params': {
                    'model_name': 'efficientnet_b4',
                    'num_classes': 2,
                    'pretrained': False,
                    'dropout_rate': 0.5
                }
            }
        
        # Create model
        model = create_model(config)
        
        # Load trained weights
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
        
        model.to(device)
        model.eval()
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def initialize_preprocessor():
    """Initialize data preprocessor"""
    global preprocessor
    
    try:
        preprocessor = DataPreprocessor(
            input_size=(224, 224),
            face_detector='mtcnn',
            max_faces_per_video=10,
            frame_rate=1
        )
        logger.info("Data preprocessor initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing preprocessor: {e}")
        return False

def predict_image(image_path: str) -> Dict[str, Any]:
    """
    Predict deepfake for a single image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Prediction results
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Detect faces
        faces = preprocessor.detect_faces(image_array)
        
        if not faces:
            return {
                'success': False,
                'error': 'No faces detected in the image'
            }
        
        predictions = []
        face_images = []
        
        for i, face in enumerate(faces):
            # Crop face
            face_crop = preprocessor.crop_face(image_array, face['box'])
            
            if face_crop is None:
                continue
            
            # Preprocess face
            processed_face = preprocessor.preprocess_image(face_crop)
            
            # Add batch dimension
            processed_face = processed_face.unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(processed_face)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Convert face crop to base64 for display
            face_pil = Image.fromarray(face_crop)
            face_buffer = io.BytesIO()
            face_pil.save(face_buffer, format='JPEG')
            face_base64 = base64.b64encode(face_buffer.getvalue()).decode()
            
            predictions.append({
                'face_id': i,
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': confidence,
                'fake_probability': probabilities[0][1].item(),
                'real_probability': probabilities[0][0].item(),
                'face_image': face_base64,
                'face_box': face['box']
            })
            
            face_images.append(face_crop)
        
        if not predictions:
            return {
                'success': False,
                'error': 'No valid faces found for analysis'
            }
        
        # Overall prediction (average of all faces)
        avg_fake_prob = np.mean([p['fake_probability'] for p in predictions])
        overall_prediction = 'Fake' if avg_fake_prob > 0.5 else 'Real'
        
        return {
            'success': True,
            'overall_prediction': overall_prediction,
            'overall_confidence': avg_fake_prob if overall_prediction == 'Fake' else 1 - avg_fake_prob,
            'face_predictions': predictions,
            'num_faces': len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def predict_video(video_path: str) -> Dict[str, Any]:
    """
    Predict deepfake for a video
    
    Args:
        video_path: Path to video file
        
    Returns:
        Prediction results
    """
    try:
        # Extract frames
        frames = preprocessor.extract_frames_from_video(video_path)
        
        if not frames:
            return {
                'success': False,
                'error': 'Could not extract frames from video'
            }
        
        frame_predictions = []
        all_faces = []
        
        for frame_idx, frame in enumerate(frames):
            # Detect faces in frame
            faces = preprocessor.detect_faces(frame)
            
            if not faces:
                continue
            
            frame_faces = []
            
            for face_idx, face in enumerate(faces):
                # Crop face
                face_crop = preprocessor.crop_face(frame, face['box'])
                
                if face_crop is None:
                    continue
                
                # Preprocess face
                processed_face = preprocessor.preprocess_image(face_crop)
                processed_face = processed_face.unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(processed_face)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                frame_faces.append({
                    'face_id': face_idx,
                    'prediction': 'Fake' if prediction == 1 else 'Real',
                    'confidence': confidence,
                    'fake_probability': probabilities[0][1].item(),
                    'real_probability': probabilities[0][0].item(),
                    'face_box': face['box']
                })
                
                all_faces.append(face_crop)
            
            if frame_faces:
                frame_predictions.append({
                    'frame_id': frame_idx,
                    'faces': frame_faces
                })
        
        if not frame_predictions:
            return {
                'success': False,
                'error': 'No faces detected in video frames'
            }
        
        # Calculate overall prediction
        all_fake_probs = []
        for frame_pred in frame_predictions:
            for face_pred in frame_pred['faces']:
                all_fake_probs.append(face_pred['fake_probability'])
        
        avg_fake_prob = np.mean(all_fake_probs)
        overall_prediction = 'Fake' if avg_fake_prob > 0.5 else 'Real'
        
        return {
            'success': True,
            'overall_prediction': overall_prediction,
            'overall_confidence': avg_fake_prob if overall_prediction == 'Fake' else 1 - avg_fake_prob,
            'frame_predictions': frame_predictions,
            'num_frames': len(frame_predictions),
            'total_faces': len(all_faces)
        }
        
    except Exception as e:
        logger.error(f"Error predicting video: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {filename}")
        
        # Determine file type and predict
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
            # Video file
            result = predict_video(file_path)
        else:
            # Image file
            result = predict_image(file_path)
        
        # Add file info to result
        result['filename'] = filename
        result['file_type'] = 'video' if file_extension in ['mp4', 'avi', 'mov', 'mkv', 'wmv'] else 'image'
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'device': str(device) if device else None
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get model summary
        from src.model import get_model_summary
        summary = get_model_summary(model)
        
        return jsonify({
            'model_type': summary['model_type'],
            'parameters': summary['trainable_parameters'],
            'parameters_mb': summary['trainable_parameters_mb'],
            'device': str(device)
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'file_path' not in data:
            return jsonify({'error': 'No file path provided'}), 400
        
        file_path = data['file_path']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Determine file type
        file_extension = file_path.rsplit('.', 1)[1].lower()
        
        if file_extension in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
            result = predict_video(file_path)
        else:
            result = predict_image(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in API predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    """Application factory"""
    # Load model and preprocessor
    if not load_model():
        logger.error("Failed to load model")
        return None
    
    if not initialize_preprocessor():
        logger.error("Failed to initialize preprocessor")
        return None
    
    logger.info("Web application initialized successfully")
    return app

if __name__ == '__main__':
    # Initialize application
    app = create_app()
    
    if app:
        # Run the application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        logger.error("Failed to initialize application")
        sys.exit(1)
