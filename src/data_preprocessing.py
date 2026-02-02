"""
HERIDAL Dataset Preprocessing Pipeline
Based on QLDNet paper methodology (Section 3.1)

This script replicates the exact data processing steps:
1. VOC XML to YOLO format conversion
2. Image resizing from 4000×3000 to 1536×1536
3. CLAHE enhancement (64×64 tiles)
4. Train/Val/Test split
5. Data augmentation setup
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import json
from typing import Tuple, List, Dict


class HERIDALPreprocessor:
    """
    Preprocessor for HERIDAL dataset following QLDNet paper methodology
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 target_size: Tuple[int, int] = (1536, 1536),
                 clahe_tile_size: int = 64,
                 val_split: float = 0.1):
        """
        Args:
            input_dir: Path to raw HERIDAL dataset
            output_dir: Path to save processed dataset
            target_size: Target image size (default: 1536×1536 as in paper)
            clahe_tile_size: CLAHE tile size (default: 64×64 as in paper)
            val_split: Validation split ratio (default: 0.1 as in paper)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.clahe_tile_size = clahe_tile_size
        self.val_split = val_split
        
        # Create output directories
        self.create_directory_structure()
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        
    def create_directory_structure(self):
        """Create YOLO-format directory structure"""
        dirs = [
            'images/train',
            'images/val',
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Created directory structure at {self.output_dir}")
    
    def parse_voc_xml(self, xml_path: Path) -> List[Dict]:
        """
        Parse Pascal VOC XML annotation file
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            List of annotation dictionaries
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        annotations = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            # Skip if not 'person' class (HERIDAL only has 'person')
            if name.lower() != 'person':
                continue
                
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            annotations.append({
                'class': 0,  # 'person' is class 0 in YOLO
                'bbox': [xmin, ymin, xmax, ymax],
                'img_width': img_width,
                'img_height': img_height
            })
            
        return annotations
    
    def voc_to_yolo(self, annotations: List[Dict], 
                    img_width: int, img_height: int) -> List[str]:
        """
        Convert VOC format to YOLO format
        Following equations from the paper
        
        Args:
            annotations: List of VOC annotations
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of YOLO format strings
        """
        yolo_lines = []
        
        for ann in annotations:
            xmin, ymin, xmax, ymax = ann['bbox']
            
            # Apply equations from paper (Section 4.1.1)
            # cx = (xmin + xmax) / (2 × width)
            cx = (xmin + xmax) / (2.0 * img_width)
            
            # cy = (ymin + ymax) / (2 × height)
            cy = (ymin + ymax) / (2.0 * img_height)
            
            # w = (xmax - xmin) / width
            w = (xmax - xmin) / img_width
            
            # h = (ymax - ymin) / height
            h = (ymax - ymin) / img_height
            
            # YOLO format: class_id cx cy w h (normalized 0-1)
            yolo_line = f"{ann['class']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)
            
        return yolo_lines
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        with 64×64 pixel tiles as specified in paper (Section 3.1)
        
        Args:
            image: Input BGR image
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size (1536×1536) as in paper
        Uses letterbox resizing to maintain aspect ratio
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor (maintain aspect ratio)
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas.fill(114)  # Gray padding
        
        # Calculate padding
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        
        # Place resized image on canvas
        canvas[top:top+new_h, left:left+new_w] = resized
        
        return canvas, scale, (left, top)
    
    def adjust_bbox_for_resize(self, bbox: List[float], 
                               scale: float, 
                               offset: Tuple[int, int]) -> List[float]:
        """
        Adjust bounding box coordinates after letterbox resize
        
        Args:
            bbox: [xmin, ymin, xmax, ymax] in original image
            scale: Scaling factor applied
            offset: (left, top) padding offset
            orig_size: Original image (width, height)
            
        Returns:
            Adjusted bbox coordinates
        """
        xmin, ymin, xmax, ymax = bbox
        left, top = offset
        
        # Apply scaling and offset
        xmin = xmin * scale + left
        ymin = ymin * scale + top
        xmax = xmax * scale + left
        ymax = ymax * scale + top
        
        return [xmin, ymin, xmax, ymax]
    
    def process_split(self, split: str, file_list: List[str]):
        """
        Process a data split (train/val/test)
        
        Args:
            split: 'train', 'val', or 'test'
            file_list: List of image filenames to process
        """
        print(f"\nProcessing {split} split ({len(file_list)} images)...")
        
        stats = {
            'total_images': len(file_list),
            'total_annotations': 0,
            'avg_objects_per_image': 0,
            'skipped_images': 0
        }
        
        for img_name in tqdm(file_list, desc=f"Processing {split}"):
            try:
                # Construct paths
                img_path = self.input_dir / 'JPEGImages' / img_name
                xml_name = img_name.replace('.jpg', '.xml').replace('.png', '.xml')
                xml_path = self.input_dir / 'Annotations' / xml_name
                
                # Check if files exist
                if not img_path.exists() or not xml_path.exists():
                    print(f"Warning: Missing files for {img_name}")
                    stats['skipped_images'] += 1
                    continue
                
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: Could not read {img_path}")
                    stats['skipped_images'] += 1
                    continue
                                
                # Parse annotations
                annotations = self.parse_voc_xml(xml_path)
                
                if len(annotations) == 0:
                    print(f"Warning: No annotations for {img_name}")
                    stats['skipped_images'] += 1
                    continue
                
                stats['total_annotations'] += len(annotations)
                
                # Apply CLAHE enhancement
                enhanced = self.apply_clahe(image)
                
                # Resize image
                resized, scale, offset = self.resize_image(enhanced)
                
                # Adjust bounding boxes
                adjusted_annotations = []
                for ann in annotations:
                    adjusted_bbox = self.adjust_bbox_for_resize(
                        ann['bbox'], scale, offset
                    )
                    adjusted_annotations.append({
                        'class': ann['class'],
                        'bbox': adjusted_bbox,
                        'img_width': self.target_size[0],
                        'img_height': self.target_size[1]
                    })
                
                # Convert to YOLO format
                yolo_labels = self.voc_to_yolo(
                    adjusted_annotations,
                    self.target_size[0],
                    self.target_size[1]
                )
                
                # Save processed image
                output_img_path = self.output_dir / 'images' / split / img_name
                cv2.imwrite(str(output_img_path), resized)
                
                # Save YOLO labels
                label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                output_label_path = self.output_dir / 'labels' / split / label_name
                
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                stats['skipped_images'] += 1
        
        # Calculate statistics
        valid_images = stats['total_images'] - stats['skipped_images']
        if valid_images > 0:
            stats['avg_objects_per_image'] = stats['total_annotations'] / valid_images
        
        print(f"\n{split.upper()} Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Valid images: {valid_images}")
        print(f"  Skipped images: {stats['skipped_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Avg objects per image: {stats['avg_objects_per_image']:.2f}")
        
        return stats

    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        yaml_content = f"""# HERIDAL Dataset Configuration
# Based on QLDNet paper preprocessing

path: {self.output_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
names:
  0: person

# Dataset info
nc: 1  # number of classes
original_resolution: [4000, 3000]
processed_resolution: {list(self.target_size)}
clahe_tile_size: {self.clahe_tile_size}
validation_split: {self.val_split}

# Paper reference
reference: "QLDNet: A Lightweight and Efficient Network for High-Robustness Aerial Human Detection"
dataset: "HERIDAL - Human Detection in Aerial Images Dataset"
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        print(f"\n✓ Created dataset.yaml at {yaml_path}")
    
    def process_dataset(self):
        """
        Main processing pipeline
        """
        print("="*60)
        print("HERIDAL Dataset Preprocessing")
        print("Following QLDNet paper methodology")
        print("="*60)
        
        # Get file lists from Annotations directory
        annotations_dir = self.input_dir / 'Annotations'
        
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
        
        # Identify train, val and test files based on split txt files
        def read_split_file(split_path: Path) -> List[str]:
            with open(split_path, 'r') as f:
                files = [line.strip() + '.jpg' for line in f if line.strip()]
            return files
        train_files = read_split_file(self.input_dir / 'ImageSets/train.txt')
        val_files   = read_split_file(self.input_dir / 'ImageSets/val.txt')
        test_files  = read_split_file(self.input_dir / 'ImageSets/test.txt')

        print(f"\nDataset split (based on official txt files):")
        print(f"  Training files: {len(train_files)}")
        print(f"  Validation files: {len(val_files)}")
        print(f"  Test files: {len(test_files)}")

        if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
            raise ValueError("Could not find train/val/test split. Check dataset structure.")
        
        # Process each split
        train_stats = self.process_split('train', train_files)
        val_stats = self.process_split('val', val_files)
        test_stats = self.process_split('test', test_files)
        
        # Create dataset.yaml
        self.create_dataset_yaml()
        
        # Save processing report
        self.save_report(train_stats, val_stats, test_stats)
        
        print("\n" + "="*60)
        print("✓ Preprocessing completed successfully!")
        print("="*60)
    
    def save_report(self, train_stats: Dict, val_stats: Dict, test_stats: Dict):
        """Save preprocessing report"""
        report = {
            'preprocessing_config': {
                'target_size': self.target_size,
                'clahe_tile_size': self.clahe_tile_size,
                'validation_split': self.val_split,
                'paper_reference': 'QLDNet (Section 3.1)'
            },
            'statistics': {
                'train': train_stats,
                'val': val_stats,
                'test': test_stats
            }
        }
        
        report_path = self.output_dir / 'preprocessing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n✓ Saved preprocessing report to {report_path}")
