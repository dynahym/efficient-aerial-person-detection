"""
Training Scripts for Lightweight Object Detection Models on HERIDAL Dataset

Models implemented:
1. YOLOv8-n (lightweight YOLO)

Training configuration follows QLDNet paper:
- Input size: 1536×1536
- Epochs: 300
- Optimizer: Adam
- Learning rate: 0.01 → 0.001 (CosineAnnealing)
- Weight decay: 0.0005
- Batch size: 1 (adjustable based on GPU memory)
- Random seed: 42
- Loss weights: Box=0.7, Classification=0.3
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HERIDALDataset(Dataset):
    """
    HERIDAL Dataset loader for PyTorch
    """
    
    def __init__(self, 
                 data_dir: Path,
                 split: str = 'train',
                 img_size: int = 1536):
        """
        Args:
            data_dir: Root directory of processed HERIDAL dataset
            split: 'train', 'val', or 'test'
            img_size: Image size
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # Get image and label paths
        self.img_dir = self.data_dir / 'images' / split
        self.label_dir = self.data_dir / 'labels' / split
        
        # Get all image files
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + 
                               list(self.img_dir.glob('*.png')))
        
        print(f"Loaded {len(self.img_files)} images for {split} split")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        bboxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        bboxes.append([class_id, cx, cy, w, h])
        
        bboxes = np.array(bboxes) if bboxes else np.zeros((0, 5))
        
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        bboxes = torch.from_numpy(bboxes).float()
        
        return image, bboxes


class YOLOv8Trainer:
    """
    YOLOv8-n trainer for HERIDAL dataset using Ultralytics
    """
    
    def __init__(self, data_yaml: str, output_dir: str):
        """
        Args:
            data_yaml: Path to dataset YAML file
            output_dir: Output directory for results
        """
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import ultralytics
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
    
    def train(self, 
              epochs: int = 300,
              batch_size: int = 2,
              img_size: int = 1536,
              device: str = "cuda"):
        """
        Train YOLOv8-n model on HERIDAL dataset
        """
        print("\n" + "="*60)
        print("Training YOLOv8-n on HERIDAL Dataset")
        print("="*60)
        
        # Initialize YOLOv8n pretrained weights
        model = self.YOLO('yolov8n.pt')
        
        # Training configuration
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            
            # Optimizer
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            weight_decay=5e-4,
            
            # Loss weights
            box=0.7,
            cls=0.3,
            
            # Augmentations
            mosaic=1.0,       # Keep mosaic early, YOLO disables late automatically
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,      # Disable rotation
            translate=0.0,    # Disable translation
            scale=0.0,        # Disable scaling
            shear=0.0,        # Disable shear
            perspective=0.0,  # Disable perspective
            fliplr=0.0,       # Disable horizontal flip
            flipud=0.0,       # Disable vertical flip
            
            # Photometric augmentation only
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            
            # Misc
            seed=42,
            workers=8,
            project=str(self.output_dir),
            name='yolov8n_heridal',
            exist_ok=True,
            amp=True,          # Mixed precision for faster training
            
            # Save settings
            save=True,
            save_period=10,
            
            # Validation
            val=True,
            plots=True
        )
        
        print("\n✓ YOLOv8-n training completed")
        print(f"Results saved to: {self.output_dir / 'yolov8n_heridal'}")
        return results
    
    def evaluate(self, weights: str, data_yaml: str = None, batch_size: int = 1, device: str = "cuda"):
        """
        Evaluate trained YOLOv8-n model
        """
        if data_yaml is None:
            data_yaml = self.data_yaml
            
        model = self.YOLO(weights)
        results = model.val(
            data=data_yaml,
            imgsz=1536,
            batch=batch_size,
            device=device
        )
        return results
