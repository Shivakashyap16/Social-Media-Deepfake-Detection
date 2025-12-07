"""
Data Preprocessing Module for Deepfake Detection

This module handles:
1. Video frame extraction
2. Face detection and cropping
3. Image preprocessing and augmentation
4. Dataset organization and splitting

Author: Deepfake Detection Team
Date: 2024
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Face detection
# from mtcnn import MTCNN  # Temporarily disabled due to TensorFlow dependency
# import dlib  # Temporarily disabled

# Image processing
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utilities
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Main class for preprocessing deepfake detection data
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 face_detector: str = 'mtcnn',
                 max_faces_per_video: int = 10,
                 frame_rate: int = 1):
        """
        Initialize the data preprocessor
        
        Args:
            input_size: Target size for processed images (height, width)
            face_detector: Face detection method ('mtcnn' or 'dlib')
            max_faces_per_video: Maximum number of faces to extract per video
            frame_rate: Number of frames to extract per second
        """
        self.input_size = input_size
        self.face_detector = face_detector
        self.max_faces_per_video = max_faces_per_video
        self.frame_rate = frame_rate
        
        # Initialize face detector (simplified for demo)
        if face_detector == 'mtcnn':
            # self.detector = MTCNN(
            #     min_face_size=20,
            #     scale_factor=0.709,
            #     post_process=False
            # )
            self.detector = None  # Temporarily disabled
            logger.warning("MTCNN face detection is temporarily disabled. Using simple image processing instead.")
        elif face_detector == 'dlib':
            # self.detector = dlib.get_frontal_face_detector()
            self.detector = None  # Temporarily disabled
            logger.warning("DLib face detection is temporarily disabled. Using simple image processing instead.")
        else:
            raise ValueError(f"Unsupported face detector: {face_detector}")
        
        # Initialize data augmentation
        self.augmentation = self._create_augmentation_pipeline()
        
        logger.info(f"DataPreprocessor initialized with {face_detector} detector")
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline"""
        # Simplified augmentation pipeline for demo
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),  # Changed from A.Flip to A.HorizontalFlip
            A.OneOf([
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frame interval based on desired frame rate
        frame_interval = int(fps / self.frame_rate)
        
        logger.info(f"Extracting frames from {video_path}")
        logger.info(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f}s duration")
        
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=min(total_frames // frame_interval, self.max_faces_per_video)) as pbar:
            while cap.isOpened() and extracted_count < self.max_faces_per_video:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image (simplified version for demo)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        if self.detector is None:
            # For demo purposes, treat the entire image as a face
            h, w = image.shape[:2]
            return [{
                'box': [0, 0, w, h],
                'confidence': 1.0
            }]
        
        if self.face_detector == 'mtcnn':
            faces = self.detector.detect_faces(image)
            return faces
        elif self.face_detector == 'dlib':
            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            dlib_faces = self.detector(gray, 1)
            
            faces = []
            for face in dlib_faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces.append({
                    'box': [x, y, w, h],
                    'confidence': 1.0  # dlib doesn't provide confidence scores
                })
            return faces
        
        return []
    
    def crop_face(self, image: np.ndarray, face_box: List[int], 
                  margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face from image with margin
        
        Args:
            image: Input image
            face_box: Face bounding box [x, y, width, height]
            margin: Margin around face as fraction of face size
            
        Returns:
            Cropped face image or None if cropping fails
        """
        try:
            x, y, w, h = face_box
            
            # Add margin
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            # Calculate crop coordinates
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            return face_crop
            
        except Exception as e:
            logger.warning(f"Failed to crop face: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed image
        """
        if augment:
            augmented = self.augmentation(image=image)
            return augmented['image']
        else:
            # Basic preprocessing without augmentation
            transform = A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
            result = transform(image=image)
            return result['image']
    
    def process_video(self, video_path: str, label: int, 
                     output_dir: str) -> List[Dict]:
        """
        Process a single video file
        
        Args:
            video_path: Path to video file
            label: Label (0 for real, 1 for fake)
            output_dir: Output directory for processed images
            
        Returns:
            List of metadata for processed images
        """
        metadata = []
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path)
        
        if not frames:
            logger.warning(f"No frames extracted from {video_path}")
            return metadata
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Detect faces
            faces = self.detect_faces(frame)
            
            if not faces:
                logger.debug(f"No faces detected in frame {frame_idx}")
                continue
            
            # Process each detected face
            for face_idx, face in enumerate(faces):
                # Crop face
                face_crop = self.crop_face(frame, face['box'])
                
                if face_crop is None:
                    continue
                
                # Generate output filename
                video_name = Path(video_path).stem
                output_filename = f"{video_name}_frame{frame_idx}_face{face_idx}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save cropped face
                face_pil = Image.fromarray(face_crop)
                face_pil.save(output_path, quality=95)
                
                # Store metadata
                metadata.append({
                    'image_path': output_path,
                    'video_path': video_path,
                    'frame_idx': frame_idx,
                    'face_idx': face_idx,
                    'label': label,
                    'confidence': face.get('confidence', 1.0),
                    'face_box': face['box']
                })
        
        return metadata
    
    def process_dataset(self, dataset_path: str, output_dir: str,
                       real_dir: str = 'real', fake_dir: str = 'fake') -> pd.DataFrame:
        """
        Process entire dataset
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for processed images
            real_dir: Subdirectory containing real videos
            fake_dir: Subdirectory containing fake videos
            
        Returns:
            DataFrame with metadata for all processed images
        """
        all_metadata = []
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Process real videos
        real_path = os.path.join(dataset_path, real_dir)
        if os.path.exists(real_path):
            logger.info(f"Processing real videos from {real_path}")
            real_metadata = self._process_video_directory(real_path, output_dir, label=0)
            all_metadata.extend(real_metadata)
        
        # Process fake videos
        fake_path = os.path.join(dataset_path, fake_dir)
        if os.path.exists(fake_path):
            logger.info(f"Processing fake videos from {fake_path}")
            fake_metadata = self._process_video_directory(fake_path, output_dir, label=1)
            all_metadata.extend(fake_metadata)
        
        # Create DataFrame
        df = pd.DataFrame(all_metadata)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.csv')
        df.to_csv(metadata_path, index=False)
        
        logger.info(f"Dataset processing complete. Total images: {len(df)}")
        logger.info(f"Real images: {len(df[df['label'] == 0])}")
        logger.info(f"Fake images: {len(df[df['label'] == 1])}")
        
        return df
    
    def _process_video_directory(self, video_dir: str, output_dir: str, 
                                label: int) -> List[Dict]:
        """Process all videos in a directory"""
        metadata = []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} video files in {video_dir}")
        
        for video_file in tqdm(video_files, desc=f"Processing {label} videos"):
            try:
                video_metadata = self.process_video(
                    str(video_file), label, output_dir
                )
                metadata.extend(video_metadata)
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                continue
        
        return metadata
    
    def split_dataset(self, metadata_df: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets
        
        Args:
            metadata_df: DataFrame with image metadata
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Stratified split to maintain class balance
        train_df, temp_df = train_test_split(
            metadata_df, 
            test_size=(val_ratio + test_ratio),
            stratify=metadata_df['label'],
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['label'],
            random_state=42
        )
        
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        # Save splits
        for split_name, split_df in splits.items():
            split_path = os.path.join(os.path.dirname(metadata_df.iloc[0]['image_path']), 
                                    f'{split_name}_split.csv')
            split_df.to_csv(split_path, index=False)
            logger.info(f"{split_name.capitalize()} set: {len(split_df)} images")
        
        return splits
    
    def create_data_loader_config(self, splits: Dict[str, pd.DataFrame],
                                 output_dir: str) -> Dict:
        """
        Create configuration for data loaders
        
        Args:
            splits: Dictionary with train/validation/test DataFrames
            output_dir: Output directory
            
        Returns:
            Configuration dictionary
        """
        config = {
            'data_dir': output_dir,
            'input_size': self.input_size,
            'batch_size': 32,
            'num_workers': 4,
            'splits': {}
        }
        
        for split_name, split_df in splits.items():
            config['splits'][split_name] = {
                'csv_path': os.path.join(output_dir, f'{split_name}_split.csv'),
                'num_samples': len(split_df),
                'num_real': len(split_df[split_df['label'] == 0]),
                'num_fake': len(split_df[split_df['label'] == 1])
            }
        
        # Save configuration
        config_path = os.path.join(output_dir, 'data_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def visualize_dataset(self, metadata_df: pd.DataFrame, output_dir: str):
        """
        Create visualizations of the dataset
        
        Args:
            metadata_df: DataFrame with image metadata
            output_dir: Output directory for visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(10, 6))
        class_counts = metadata_df['label'].value_counts()
        plt.pie(class_counts.values, labels=['Real', 'Fake'], autopct='%1.1f%%')
        plt.title('Dataset Class Distribution')
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close()
        
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(metadata_df['confidence'], bins=50, alpha=0.7)
        plt.xlabel('Face Detection Confidence')
        plt.ylabel('Frequency')
        plt.title('Face Detection Confidence Distribution')
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()
        
        # Sample images
        self._create_sample_grid(metadata_df, output_dir)
    
    def _create_sample_grid(self, metadata_df: pd.DataFrame, output_dir: str, 
                           num_samples: int = 16):
        """Create a grid of sample images"""
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        # Sample equal numbers of real and fake images
        real_samples = metadata_df[metadata_df['label'] == 0].sample(n=num_samples//2)
        fake_samples = metadata_df[metadata_df['label'] == 1].sample(n=num_samples//2)
        samples = pd.concat([real_samples, fake_samples]).sample(frac=1)
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= num_samples:
                break
            
            try:
                img = Image.open(row['image_path'])
                axes[idx].imshow(img)
                axes[idx].set_title(f"{'Real' if row['label'] == 0 else 'Fake'}")
                axes[idx].axis('off')
            except Exception as e:
                logger.warning(f"Could not load image {row['image_path']}: {e}")
                axes[idx].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_images.png'))
        plt.close()


def main():
    """Example usage of the DataPreprocessor"""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        input_size=(224, 224),
        face_detector='mtcnn',
        max_faces_per_video=10,
        frame_rate=1
    )
    
    # Example paths (you'll need to adjust these)
    dataset_path = "path/to/your/dataset"
    output_dir = "path/to/processed/data"
    
    # Process dataset
    metadata_df = preprocessor.process_dataset(dataset_path, output_dir)
    
    # Split dataset
    splits = preprocessor.split_dataset(metadata_df)
    
    # Create configuration
    config = preprocessor.create_data_loader_config(splits, output_dir)
    
    # Create visualizations
    preprocessor.visualize_dataset(metadata_df, output_dir)
    
    print("Data preprocessing complete!")


if __name__ == "__main__":
    main()
