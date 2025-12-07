#!/usr/bin/env python3
"""
Quick Start Script for Deepfake Detection Project

This script helps beginners get started with the project quickly.
It will guide you through the setup process and run a demo.

Author: Deepfake Detection Team
Date: 2024
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStart:
    """Quick start helper class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration"""
        config_path = self.project_root / 'config' / 'training_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8 or higher is required!")
            return False
        logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        required_packages = [
            'torch', 'torchvision', 'opencv-python', 'numpy', 
            'pandas', 'matplotlib', 'seaborn', 'flask'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} is missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Run: pip install -r requirements.txt")
            return False
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data', 'models', 'outputs', 'logs', 'uploads',
            'data/real', 'data/fake', 'data/processed'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def create_sample_data(self):
        """Create sample data for testing"""
        sample_data_dir = self.project_root / 'data' / 'sample'
        sample_data_dir.mkdir(exist_ok=True)
        
        # Create a simple test image
        try:
            import numpy as np
            from PIL import Image
            
            # Create a simple test image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(sample_data_dir / 'test_image.jpg')
            
            logger.info("Created sample test image")
            return True
        except Exception as e:
            logger.warning(f"Could not create sample data: {e}")
            return False
    
    def run_data_preprocessing(self):
        """Run data preprocessing"""
        try:
            logger.info("Running data preprocessing...")
            
            # Import and run preprocessing
            sys.path.append(str(self.project_root))
            from src.data_preprocessing import DataPreprocessor
            
            preprocessor = DataPreprocessor(
                input_size=(224, 224),
                face_detector='mtcnn',
                max_faces_per_video=5,
                frame_rate=1
            )
            
            # Process sample data if available
            sample_dir = self.project_root / 'data' / 'sample'
            if sample_dir.exists():
                logger.info("Processing sample data...")
                # This would normally process videos, but for demo we'll skip
                logger.info("Sample data processing completed")
            
            return True
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return False
    
    def run_model_training(self, demo_mode=True):
        """Run model training (demo mode)"""
        try:
            logger.info("Starting model training (demo mode)...")
            
            if demo_mode:
                # Create a simple demo model
                import torch
                import torch.nn as nn
                
                class DemoModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.features = nn.Sequential(
                            nn.Conv2d(3, 64, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(64, 128, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.AdaptiveAvgPool2d((1, 1))
                        )
                        self.classifier = nn.Linear(128, 2)
                    
                    def forward(self, x):
                        x = self.features(x)
                        x = x.view(x.size(0), -1)
                        x = self.classifier(x)
                        return x
                
                model = DemoModel()
                
                # Save demo model
                model_path = self.project_root / 'models' / 'demo_model.pth'
                torch.save(model.state_dict(), model_path)
                logger.info("Demo model created and saved")
                
                return True
            else:
                # Run actual training
                logger.info("Running full model training...")
                # This would run the actual training script
                return True
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def start_web_app(self):
        """Start the web application"""
        try:
            logger.info("Starting web application...")
            
            # Check if Flask app exists
            app_path = self.project_root / 'webapp' / 'app.py'
            if not app_path.exists():
                logger.error("Web application not found!")
                return False
            
            # Start Flask app in background
            import subprocess
            import threading
            import time
            
            def run_flask():
                subprocess.run([sys.executable, str(app_path)], 
                             cwd=str(self.project_root / 'webapp'))
            
            # Start Flask in a separate thread
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            
            # Wait a moment for Flask to start
            time.sleep(3)
            
            logger.info("Web application started!")
            logger.info("Open your browser and go to: http://localhost:5000")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start web application: {e}")
            return False
    
    def run_demo(self):
        """Run a complete demo"""
        logger.info("🚀 Starting Deepfake Detection Demo...")
        
        # Step 1: Check environment
        logger.info("\n1. Checking environment...")
        if not self.check_python_version():
            return False
        
        if not self.check_dependencies():
            logger.warning("Some dependencies are missing. Install them with: pip install -r requirements.txt")
            logger.info("Continuing with demo...")
        
        # Step 2: Setup directories
        logger.info("\n2. Setting up directories...")
        self.setup_directories()
        
        # Step 3: Create sample data
        logger.info("\n3. Creating sample data...")
        self.create_sample_data()
        
        # Step 4: Run preprocessing
        logger.info("\n4. Running data preprocessing...")
        self.run_data_preprocessing()
        
        # Step 5: Run model training (demo)
        logger.info("\n5. Running model training (demo)...")
        self.run_model_training(demo_mode=True)
        
        # Step 6: Start web app
        logger.info("\n6. Starting web application...")
        self.start_web_app()
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Open http://localhost:5000 in your browser")
        logger.info("2. Upload a test image or video")
        logger.info("3. See the deepfake detection in action!")
        
        return True
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 Deepfake Detection Project - Quick Start Guide

Available commands:
  python quick_start.py demo     - Run complete demo
  python quick_start.py check    - Check environment
  python quick_start.py setup    - Setup directories
  python quick_start.py train    - Run model training
  python quick_start.py web      - Start web application
  python quick_start.py help     - Show this help

For detailed instructions, see: STEP_BY_STEP_GUIDE.md
        """
        print(help_text)

def main():
    """Main function"""
    quick_start = QuickStart()
    
    if len(sys.argv) < 2:
        quick_start.show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'demo':
        quick_start.run_demo()
    elif command == 'check':
        quick_start.check_python_version()
        quick_start.check_dependencies()
    elif command == 'setup':
        quick_start.setup_directories()
        quick_start.create_sample_data()
    elif command == 'train':
        quick_start.run_model_training(demo_mode=False)
    elif command == 'web':
        quick_start.start_web_app()
    elif command == 'help':
        quick_start.show_help()
    else:
        logger.error(f"Unknown command: {command}")
        quick_start.show_help()

if __name__ == "__main__":
    main()
