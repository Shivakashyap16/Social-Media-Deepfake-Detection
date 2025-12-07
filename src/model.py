"""
Deepfake Detection Model Architectures

This module contains various model architectures for deepfake detection:
1. CNN-based models (ResNet, EfficientNet)
2. Vision Transformer models
3. Custom architectures
4. Ensemble models

Author: Deepfake Detection Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict, Any
import timm
# from transformers import ViTForImageClassification, ViTConfig  # Commented out to avoid TensorFlow dependency
import numpy as np

class DeepfakeDetectionModel(nn.Module):
    """
    Base class for deepfake detection models
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(DeepfakeDetectionModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_features(self, x):
        """Extract features from the model"""
        raise NotImplementedError


class EfficientNetModel(DeepfakeDetectionModel):
    """
    EfficientNet-based deepfake detection model
    """
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b4',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super(EfficientNetModel, self).__init__(num_classes, pretrained)
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features from the backbone"""
        return self.backbone(x)


class ResNetModel(DeepfakeDetectionModel):
    """
    ResNet-based deepfake detection model
    """
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super(ResNetModel, self).__init__(num_classes, pretrained)
        
        # Load pretrained ResNet
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the original classifier
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features from the backbone"""
        return self.backbone(x)


# class VisionTransformerModel(DeepfakeDetectionModel):
#     """
#     Vision Transformer-based deepfake detection model
#     """
#     
#     def __init__(self, 
#                  model_name: str = 'vit_base_patch16_224',
#                  num_classes: int = 2,
#                  pretrained: bool = True,
#                  dropout_rate: float = 0.5):
#         super(VisionTransformerModel, self).__init__(num_classes, pretrained)
#         
#         # Load pretrained ViT
#         self.backbone = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             num_classes=0  # Remove classifier
#         )
#         
#         # Get feature dimension
#         feature_dim = self.backbone.num_features
#         
#         # Add custom classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(feature_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, num_classes)
#         )
#         
#         # Initialize classifier weights
#         self._initialize_weights()
#     
#     def _initialize_weights(self):
#         """Initialize classifier weights"""
#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#     
#     def forward(self, x):
#         features = self.backbone(x)
#         output = self.classifier(features)
#         return output
#     
#     def get_features(self, x):
#         """Extract features from the backbone"""
#         return self.backbone(x)


class CustomCNNModel(DeepfakeDetectionModel):
    """
    Custom CNN architecture for deepfake detection
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        super(CustomCNNModel, self).__init__(num_classes, pretrained=False)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Fifth block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        output = self.classifier(flattened)
        return output
    
    def get_features(self, x):
        """Extract features from the model"""
        features = self.features(x)
        pooled = self.global_pool(features)
        return torch.flatten(pooled, 1)


class EnsembleModel(DeepfakeDetectionModel):
    """
    Ensemble model combining multiple architectures
    """
    
    def __init__(self, 
                 model_configs: list,
                 num_classes: int = 2,
                 ensemble_method: str = 'average'):
        super(EnsembleModel, self).__init__(num_classes, pretrained=False)
        
        self.ensemble_method = ensemble_method
        self.models = nn.ModuleList()
        
        # Create individual models
        for config in model_configs:
            model_type = config['type']
            model_params = config.get('params', {})
            
            if model_type == 'efficientnet':
                model = EfficientNetModel(**model_params)
            elif model_type == 'resnet':
                model = ResNetModel(**model_params)
            elif model_type == 'vit':
                # VisionTransformerModel is temporarily disabled due to TensorFlow dependency issues
                raise ValueError("Vision Transformer model is temporarily disabled. Please use 'efficientnet' or 'resnet' instead.")
            elif model_type == 'custom_cnn':
                model = CustomCNNModel(**model_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.models.append(model)
        
        # Ensemble weights (can be learned or fixed)
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))
        
        # Softmax for weight normalization
        self.weight_softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=0)
        
        # Apply ensemble weights
        weights = self.weight_softmax(self.ensemble_weights)
        weighted_outputs = stacked_outputs * weights.unsqueeze(1).unsqueeze(2)
        
        # Combine outputs
        if self.ensemble_method == 'average':
            final_output = torch.mean(weighted_outputs, dim=0)
        elif self.ensemble_method == 'weighted_average':
            final_output = torch.sum(weighted_outputs, dim=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        return final_output
    
    def get_features(self, x):
        """Extract features from all models"""
        features = []
        for model in self.models:
            feature = model.get_features(x)
            features.append(feature)
        return torch.cat(features, dim=1)


class AttentionModule(nn.Module):
    """
    Attention module for focusing on important regions
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class AttentionEfficientNetModel(EfficientNetModel):
    """
    EfficientNet with attention mechanism
    """
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b4',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super(AttentionEfficientNetModel, self).__init__(
            model_name, num_classes, pretrained, dropout_rate
        )
        
        # Add attention module
        feature_dim = self.backbone.num_features
        self.attention = AttentionModule(feature_dim)
        
        # Modify classifier to work with attention
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        # Apply attention
        attended_features = self.attention(features.unsqueeze(-1).unsqueeze(-1))
        attended_features = attended_features.squeeze(-1).squeeze(-1)
        output = self.classifier(attended_features)
        return output


def create_model(model_config: Dict[str, Any]) -> DeepfakeDetectionModel:
    """
    Factory function to create models based on configuration
    
    Args:
        model_config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = model_config['type']
    
    if model_type == 'efficientnet':
        return EfficientNetModel(**model_config.get('params', {}))
    elif model_type == 'resnet':
        return ResNetModel(**model_config.get('params', {}))
    elif model_type == 'vit':
        # VisionTransformerModel is temporarily disabled due to TensorFlow dependency issues
        raise ValueError("Vision Transformer model is temporarily disabled. Please use 'efficientnet' or 'resnet' instead.")
    elif model_type == 'custom_cnn':
        return CustomCNNModel(**model_config.get('params', {}))
    elif model_type == 'attention_efficientnet':
        return AttentionEfficientNetModel(**model_config.get('params', {}))
    elif model_type == 'ensemble':
        return EnsembleModel(**model_config.get('params', {}))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
    """
    Get model summary including parameter count and memory usage
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        
    Returns:
        Model summary dictionary
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def count_parameters_total(model):
        return sum(p.numel() for p in model.parameters())
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Count parameters
    trainable_params = count_parameters(model)
    total_params = count_parameters_total(model)
    
    # Estimate memory usage
    try:
        with torch.no_grad():
            output = model(dummy_input)
            output_size = output.element_size() * output.nelement()
    except:
        output_size = 0
    
    summary = {
        'model_type': model.__class__.__name__,
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'trainable_parameters_mb': trainable_params * 4 / (1024 * 1024),  # Assuming float32
        'total_parameters_mb': total_params * 4 / (1024 * 1024),
        'output_size_mb': output_size / (1024 * 1024) if output_size > 0 else 0
    }
    
    return summary


# Example model configurations
MODEL_CONFIGS = {
    'efficientnet_b4': {
        'type': 'efficientnet',
        'params': {
            'model_name': 'efficientnet_b4',
            'num_classes': 2,
            'pretrained': True,
            'dropout_rate': 0.5
        }
    },
    'resnet50': {
        'type': 'resnet',
        'params': {
            'model_name': 'resnet50',
            'num_classes': 2,
            'pretrained': True,
            'dropout_rate': 0.5
        }
    },
    'vit_base': {
        'type': 'vit',
        'params': {
            'model_name': 'vit_base_patch16_224',
            'num_classes': 2,
            'pretrained': True,
            'dropout_rate': 0.5
        }
    },
    'custom_cnn': {
        'type': 'custom_cnn',
        'params': {
            'input_channels': 3,
            'num_classes': 2,
            'dropout_rate': 0.5
        }
    },
    'ensemble': {
        'type': 'ensemble',
        'params': {
            'model_configs': [
                {'type': 'efficientnet', 'params': {'model_name': 'efficientnet_b4'}},
                {'type': 'resnet', 'params': {'model_name': 'resnet50'}},
                {'type': 'vit', 'params': {'model_name': 'vit_base_patch16_224'}}
            ],
            'num_classes': 2,
            'ensemble_method': 'weighted_average'
        }
    }
}


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a model
    config = MODEL_CONFIGS['efficientnet_b4']
    model = create_model(config)
    model.to(device)
    
    # Get model summary
    summary = get_model_summary(model)
    print("Model Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
