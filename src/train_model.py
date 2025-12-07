"""
Model Training Script for Deepfake Detection

This script handles:
1. Model training with various optimizers and schedulers
2. Validation and evaluation
3. Model checkpointing and early stopping
4. Training visualization and logging
5. Hyperparameter optimization

Author: Deepfake Detection Team
Date: 2024
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model, MODEL_CONFIGS, get_model_summary
from src.data_preprocessing import DataPreprocessor
from utils.dataset import DeepfakeDataset
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from utils.logger import setup_logger

# Setup logging
logger = setup_logger('train_model')

class Trainer:
    """
    Main training class for deepfake detection models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model, optimizer, and scheduler
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_criterion()
        
        # Initialize logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model: {self.config['model']['type']}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
    
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
    
    def setup_model(self):
        """Initialize model"""
        model_config = self.config['model']
        self.model = create_model(model_config)
        self.model.to(self.device)
        
        # Print model summary
        summary = get_model_summary(self.model)
        logger.info("Model Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Save model summary
        with open(self.output_dir / 'model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def setup_optimizer(self):
        """Initialize optimizer"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_name = optimizer_config['name'].lower()
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer: {optimizer_name}")
    
    def setup_scheduler(self):
        """Initialize learning rate scheduler"""
        scheduler_config = self.config['training'].get('scheduler', {})
        if not scheduler_config:
            self.scheduler = None
            return
        
        scheduler_name = scheduler_config['name'].lower()
        
        if scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config['training']['epochs'])
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        logger.info(f"Scheduler: {scheduler_name}")
    
    def setup_criterion(self):
        """Initialize loss function"""
        criterion_config = self.config['training'].get('criterion', {})
        criterion_name = criterion_config.get('name', 'cross_entropy').lower()
        
        if criterion_name == 'cross_entropy':
            # Handle class imbalance with weighted loss
            if 'class_weights' in criterion_config:
                weights = torch.tensor(criterion_config['class_weights']).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == 'focal':
            from utils.losses import FocalLoss
            self.criterion = FocalLoss(
                alpha=criterion_config.get('alpha', 1.0),
                gamma=criterion_config.get('gamma', 2.0)
            )
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
        
        logger.info(f"Criterion: {criterion_name}")
    
    def setup_logging(self):
        """Setup TensorBoard logging"""
        log_dir = self.output_dir / 'logs' / 'tensorboard'
        self.writer = SummaryWriter(log_dir)
        
        # Log model graph
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
    
    def setup_data_loaders(self, data_config: Dict[str, Any]):
        """Setup data loaders"""
        # Load metadata
        train_df = pd.read_csv(data_config['splits']['train']['csv_path'])
        val_df = pd.read_csv(data_config['splits']['validation']['csv_path'])
        test_df = pd.read_csv(data_config['splits']['test']['csv_path'])
        
        # Create datasets
        self.train_dataset = DeepfakeDataset(
            train_df, 
            data_config['data_dir'],
            transform_mode='train'
        )
        self.val_dataset = DeepfakeDataset(
            val_df, 
            data_config['data_dir'],
            transform_mode='val'
        )
        self.test_dataset = DeepfakeDataset(
            test_df, 
            data_config['data_dir'],
            transform_mode='test'
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.output_dir / 'checkpoints' / 'best.pth'
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"Saved best model at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, self.current_epoch)
    
    def train(self, data_config: Dict[str, Any]):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Setup data loaders
        self.setup_data_loaders(data_config)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Log metrics
            self.log_metrics(train_metrics, 'train')
            self.log_metrics(val_metrics, 'val')
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # Print progress
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training'].get('patience', 20):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Close TensorBoard writer
        self.writer.close()
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
    
    def evaluate(self, data_config: Dict[str, Any]):
        """Evaluate model on test set"""
        logger.info("Evaluating model...")
        
        # Load best model
        best_checkpoint_path = self.output_dir / 'checkpoints' / 'best.pth'
        if best_checkpoint_path.exists():
            self.load_checkpoint(str(best_checkpoint_path))
        else:
            logger.warning("No best checkpoint found, using current model")
        
        # Setup test data loader
        test_df = pd.read_csv(data_config['splits']['test']['csv_path'])
        self.test_dataset = DeepfakeDataset(
            test_df, 
            data_config['data_dir'],
            transform_mode='test'
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        # Evaluate
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Save results
        results = {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'metrics': metrics
        }
        
        results_path = self.output_dir / 'predictions' / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        self.create_evaluation_plots(all_labels, all_predictions, all_probabilities)
        
        # Print results
        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def create_evaluation_plots(self, labels, predictions, probabilities):
        """Create evaluation plots"""
        plots_dir = self.output_dir / 'plots'
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        plot_confusion_matrix(cm, ['Real', 'Fake'], 
                            save_path=plots_dir / 'confusion_matrix.png')
        
        # ROC curve
        plot_roc_curve(labels, np.array(probabilities)[:, 1], 
                      save_path=plots_dir / 'roc_curve.png')
        
        # Prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(probabilities)[:, 1], bins=50, alpha=0.7)
        plt.xlabel('Fake Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Fake Probabilities')
        plt.savefig(plots_dir / 'prediction_distribution.png')
        plt.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate the model')
    
    args = parser.parse_args()
    
    # Load configurations
    config = load_config(args.config)
    data_config = load_config(args.data_config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    if args.evaluate_only:
        # Only evaluate
        trainer.evaluate(data_config)
    else:
        # Train and evaluate
        trainer.train(data_config)
        trainer.evaluate(data_config)


if __name__ == "__main__":
    main()
