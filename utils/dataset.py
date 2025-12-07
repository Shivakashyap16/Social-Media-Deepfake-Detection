"""
Dataset Classes for Deepfake Detection

This module contains PyTorch dataset classes for loading and preprocessing
deepfake detection data with proper transforms and augmentation.

Author: Deepfake Detection Team
Date: 2024
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import logging

logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """
    PyTorch dataset for deepfake detection
    """
    
    def __init__(self, 
                 metadata_df: pd.DataFrame,
                 data_dir: str,
                 transform_mode: str = 'train',
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize dataset
        
        Args:
            metadata_df: DataFrame with image metadata
            data_dir: Directory containing processed images
            transform_mode: 'train', 'val', or 'test'
            input_size: Target input size (height, width)
        """
        self.metadata_df = metadata_df
        self.data_dir = data_dir
        self.transform_mode = transform_mode
        self.input_size = input_size
        
        # Create transforms
        self.transforms = self._create_transforms()
        
        logger.info(f"Created {transform_mode} dataset with {len(metadata_df)} samples")
    
    def _create_transforms(self) -> A.Compose:
        """Create data transforms based on mode"""
        if self.transform_mode == 'train':
            return A.Compose([
                A.RandomResizedCrop(
                    height=self.input_size[0],
                    width=self.input_size[1],
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.25),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.IAASharpen(p=0.5),
                    A.IAAEmboss(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.RGBShift(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
                    A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.5),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                ], p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        elif self.transform_mode == 'val':
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        elif self.transform_mode == 'test':
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        else:
            raise ValueError(f"Unsupported transform mode: {self.transform_mode}")
    
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        row = self.metadata_df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.data_dir, image_path)
        
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return image, label


class VideoDataset(Dataset):
    """
    Dataset for video-based deepfake detection
    """
    
    def __init__(self, 
                 metadata_df: pd.DataFrame,
                 data_dir: str,
                 transform_mode: str = 'train',
                 input_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 16,
                 frame_interval: int = 1):
        """
        Initialize video dataset
        
        Args:
            metadata_df: DataFrame with video metadata
            data_dir: Directory containing video files
            transform_mode: 'train', 'val', or 'test'
            input_size: Target input size (height, width)
            num_frames: Number of frames to extract per video
            frame_interval: Interval between frames
        """
        self.metadata_df = metadata_df
        self.data_dir = data_dir
        self.transform_mode = transform_mode
        self.input_size = input_size
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        
        # Create transforms
        self.transforms = self._create_transforms()
        
        logger.info(f"Created {transform_mode} video dataset with {len(metadata_df)} samples")
    
    def _create_transforms(self) -> A.Compose:
        """Create data transforms for video frames"""
        if self.transform_mode == 'train':
            return A.Compose([
                A.RandomResizedCrop(
                    height=self.input_size[0],
                    width=self.input_size[1],
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                ),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.IAASharpen(p=0.5),
                ], p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of frames
        """
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.data_dir, video_path)
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            # Return black frames as fallback
            return np.zeros((self.num_frames, 3, self.input_size[0], self.input_size[1]))
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < self.num_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                if self.transforms:
                    transformed = self.transforms(image=frame_rgb)
                    frame_tensor = transformed['image']
                else:
                    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                
                frames.append(frame_tensor)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Pad with last frame if not enough frames
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, self.input_size[0], self.input_size[1]))
        
        # Stack frames
        frames_tensor = torch.stack(frames[:self.num_frames])
        
        return frames_tensor
    
    def __len__(self) -> int:
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single video sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        row = self.metadata_df.iloc[idx]
        
        # Extract frames
        video_path = row['video_path']
        frames = self.extract_frames(video_path)
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return frames, label


class MixedDataset(Dataset):
    """
    Dataset that combines image and video samples
    """
    
    def __init__(self, 
                 image_metadata_df: pd.DataFrame,
                 video_metadata_df: pd.DataFrame,
                 data_dir: str,
                 transform_mode: str = 'train',
                 input_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 16):
        """
        Initialize mixed dataset
        
        Args:
            image_metadata_df: DataFrame with image metadata
            video_metadata_df: DataFrame with video metadata
            data_dir: Directory containing data
            transform_mode: 'train', 'val', or 'test'
            input_size: Target input size (height, width)
            num_frames: Number of frames for videos
        """
        self.image_dataset = DeepfakeDataset(
            image_metadata_df, data_dir, transform_mode, input_size
        )
        self.video_dataset = VideoDataset(
            video_metadata_df, data_dir, transform_mode, input_size, num_frames
        )
        
        # Create sample type mapping
        self.sample_types = []
        for _ in range(len(image_metadata_df)):
            self.sample_types.append('image')
        for _ in range(len(video_metadata_df)):
            self.sample_types.append('video')
        
        logger.info(f"Created mixed dataset with {len(self.sample_types)} samples")
        logger.info(f"  Images: {len(image_metadata_df)}")
        logger.info(f"  Videos: {len(video_metadata_df)}")
    
    def __len__(self) -> int:
        return len(self.sample_types)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, label, sample_type)
        """
        sample_type = self.sample_types[idx]
        
        if sample_type == 'image':
            data, label = self.image_dataset[idx]
        else:
            data, label = self.video_dataset[idx]
        
        return data, label, sample_type


def create_dataloader(dataset: Dataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with common settings
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle  # Drop last batch during training
    )


def get_class_weights(metadata_df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        metadata_df: DataFrame with labels
        
    Returns:
        Class weights tensor
    """
    class_counts = metadata_df['label'].value_counts().sort_index()
    total_samples = len(metadata_df)
    
    # Calculate weights inversely proportional to class frequencies
    weights = total_samples / (len(class_counts) * class_counts.values)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    logger.info(f"Class weights: {weights.tolist()}")
    
    return weights


def visualize_dataset_samples(dataset: Dataset, 
                             num_samples: int = 16,
                             save_path: Optional[str] = None):
    """
    Visualize random samples from dataset
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
        
        data, label = dataset[idx]
        
        # Handle different data types
        if isinstance(data, torch.Tensor):
            if data.dim() == 4:  # Video frames
                # Show first frame
                img = data[0].permute(1, 2, 0).numpy()
            else:  # Single image
                img = data.permute(1, 2, 0).numpy()
        else:
            img = data
        
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label.item()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved dataset visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    # Create sample metadata
    sample_data = {
        'image_path': ['sample1.jpg', 'sample2.jpg'],
        'label': [0, 1]
    }
    metadata_df = pd.DataFrame(sample_data)
    
    # Create dataset
    dataset = DeepfakeDataset(metadata_df, '.', 'train')
    
    # Visualize samples
    visualize_dataset_samples(dataset, num_samples=4)
