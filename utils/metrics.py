"""
Metrics and Evaluation Utilities for Deepfake Detection

This module provides comprehensive evaluation metrics and visualization
functions for deepfake detection models.

Author: Deepfake Detection Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, average_precision_score
)
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: List[int], 
                     y_pred: List[int], 
                     y_prob: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_real': precision_per_class[0],
        'precision_fake': precision_per_class[1],
        'recall_real': recall_per_class[0],
        'recall_fake': recall_per_class[1],
        'f1_real': f1_per_class[0],
        'f1_fake': f1_per_class[1]
    }
    
    # ROC-AUC if probabilities are provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        if y_prob.ndim == 2:
            # Multi-class probabilities
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            # Use probability of positive class (fake) for binary metrics
            y_prob_binary = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.flatten()
        else:
            # Binary probabilities
            roc_auc = roc_auc_score(y_true, y_prob)
            y_prob_binary = y_prob
        
        # Additional metrics for binary classification
        avg_precision = average_precision_score(y_true, y_prob_binary)
        
        metrics.update({
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        })
    
    return metrics


def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int],
                         class_names: List[str] = ['Real', 'Fake'],
                         save_path: Optional[str] = None,
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: List[int], 
                   y_prob: List[float],
                   save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true: List[int], 
                               y_prob: List[float],
                               save_path: Optional[str] = None) -> None:
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Precision-Recall curve to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None) -> None:
    """
    Plot comparison of metrics across different models/configurations
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Path to save plot
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Prepare data
    model_names = list(metrics_dict.keys())
    metric_values = {metric: [] for metric in metrics_to_plot}
    
    for model_name in model_names:
        for metric in metrics_to_plot:
            value = metrics_dict[model_name].get(metric, 0.0)
            metric_values[metric].append(value)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            axes[i].bar(model_names, metric_values[metric])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
    
    # Remove extra subplot
    if len(metrics_to_plot) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics comparison to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and metrics over epochs)
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_accuracy' in history and 'val_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot precision
    if 'train_precision' in history and 'val_precision' in history:
        axes[2].plot(history['train_precision'], label='Training Precision')
        axes[2].plot(history['val_precision'], label='Validation Precision')
        axes[2].set_title('Precision')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Precision')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot recall
    if 'train_recall' in history and 'val_recall' in history:
        axes[3].plot(history['train_recall'], label='Training Recall')
        axes[3].plot(history['val_recall'], label='Validation Recall')
        axes[3].set_title('Recall')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Recall')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
    
    plt.show()


def calculate_confidence_intervals(y_true: List[int], 
                                  y_pred: List[int],
                                  confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for metrics using bootstrap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Dictionary with confidence intervals for each metric
    """
    from scipy import stats
    
    n_bootstrap = 1000
    n_samples = len(y_true)
    
    # Bootstrap samples
    bootstrap_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_boot, y_pred_boot)
        for metric in bootstrap_metrics:
            bootstrap_metrics[metric].append(metrics[metric])
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    confidence_intervals = {}
    for metric, values in bootstrap_metrics.items():
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals


def generate_classification_report(y_true: List[int], 
                                  y_pred: List[int],
                                  class_names: List[str] = ['Real', 'Fake'],
                                  save_path: Optional[str] = None) -> str:
    """
    Generate detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save report
        
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 digits=4)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved classification report to {save_path}")
    
    return report


def calculate_threshold_metrics(y_true: List[int], 
                               y_prob: List[float],
                               thresholds: List[float] = None) -> pd.DataFrame:
    """
    Calculate metrics at different probability thresholds
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to evaluate
        
    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    
    return pd.DataFrame(results)


def plot_threshold_analysis(y_true: List[int], 
                           y_prob: List[float],
                           save_path: Optional[str] = None) -> None:
    """
    Plot threshold analysis showing how metrics change with threshold
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save plot
    """
    threshold_df = calculate_threshold_metrics(y_true, y_prob)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(threshold_df['threshold'], threshold_df['accuracy'])
    plt.title('Accuracy vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(threshold_df['threshold'], threshold_df['precision'])
    plt.title('Precision vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(threshold_df['threshold'], threshold_df['recall'])
    plt.title('Recall vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(threshold_df['threshold'], threshold_df['f1_score'])
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold analysis to {save_path}")
    
    plt.show()


def evaluate_model_performance(y_true: List[int], 
                              y_pred: List[int],
                              y_prob: Optional[List[float]] = None,
                              class_names: List[str] = ['Real', 'Fake'],
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with all metrics and plots
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: Names of classes
        output_dir: Directory to save plots and reports
        
    Returns:
        Dictionary with all evaluation results
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Generate plots
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(y_true, y_pred, class_names,
                            save_path=os.path.join(output_dir, 'confusion_matrix.png'))
        
        # ROC curve
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob,
                          save_path=os.path.join(output_dir, 'roc_curve.png'))
            
            # Precision-Recall curve
            plot_precision_recall_curve(y_true, y_prob,
                                      save_path=os.path.join(output_dir, 'pr_curve.png'))
            
            # Threshold analysis
            plot_threshold_analysis(y_true, y_prob,
                                  save_path=os.path.join(output_dir, 'threshold_analysis.png'))
    
    # Generate classification report
    report = generate_classification_report(y_true, y_pred, class_names,
                                          save_path=os.path.join(output_dir, 'classification_report.txt') if output_dir else None)
    
    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(y_true, y_pred)
    
    results = {
        'metrics': metrics,
        'classification_report': report,
        'confidence_intervals': confidence_intervals
    }
    
    # Print summary
    logger.info("Model Performance Summary:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.random(1000)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Evaluate performance
    results = evaluate_model_performance(y_true, y_pred, y_prob, output_dir='./evaluation_results')
    
    print("Evaluation completed!")
