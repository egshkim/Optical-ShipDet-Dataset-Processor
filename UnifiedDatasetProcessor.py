import os
import json
import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable, Set
from enum import Enum
from PIL import Image
from pathlib import Path
import numpy as np
from utils import *
from multiprocessing import Pool
from functools import partial
import pandas as pd
import cv2


# Configuration Management
class CoordinateFormat(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

@dataclass
class ProcessingConfig:
    input_dir: str
    output_dir: str
    target_size: Tuple[int, int]
    coord_format: CoordinateFormat
    visualize: bool = False
    visualization_dir: Optional[str] = None
    
    def validate(self) -> None:
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        if self.target_size[0] <= 0 or self.target_size[1] <= 0:
            raise ValueError("Target dimensions must be positive")

# Logging Configuration
def setup_logging(log_file: Optional[str] = None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

# Dataset Validator Interface
class DatasetValidator(ABC):
    @abstractmethod
    def validate_structure(self) -> bool:
        """Validate dataset directory structure"""
        pass
    
    @abstractmethod
    def validate_files(self) -> Tuple[bool, Set[str]]:
        """Validate dataset files and return any missing files"""
        pass

# Abstract Base Processor
class BaseDatasetProcessor(ABC):
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._setup_directories()
        self.validator = self._create_validator()
        
    def _setup_directories(self) -> None:
        """Create necessary output directories"""
        for split in ['train', 'val']:
            split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
            if self.config.visualize:
                os.makedirs(os.path.join(split_dir, 'visualizations'), exist_ok=True)

    @abstractmethod
    def _create_validator(self) -> DatasetValidator:
        """Create appropriate validator for the dataset"""
        pass

    @abstractmethod
    def process(self) -> None:
        """Process the dataset"""
        pass
    
    def validate(self) -> bool:
        """Validate dataset before processing"""
        structure_valid = self.validator.validate_structure()
        files_valid, missing_files = self.validator.validate_files()
        
        if not structure_valid:
            logging.error("Dataset structure validation failed")
            return False
            
        if not files_valid:
            logging.error(f"Missing files: {missing_files}")
            return False
            
        return True

# XView Dataset Implementation
class XViewValidator(DatasetValidator):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        
    def validate_structure(self) -> bool:
        required_dirs = ['train_images', 'train_labels']
        return all(os.path.exists(os.path.join(self.input_dir, d)) for d in required_dirs)
    
    def validate_files(self) -> Tuple[bool, Set[str]]:
        missing_files = set()
        label_file = os.path.join(self.input_dir, 'train_labels', 'xView_train.geojson')
        if not os.path.exists(label_file):
            missing_files.add(label_file)
        return len(missing_files) == 0, missing_files

class XViewProcessor(BaseDatasetProcessor):
    def _create_validator(self) -> DatasetValidator:
        return XViewValidator(self.config.input_dir)
        
    def process(self) -> None:
        if not self.validate():
            raise ValueError("Dataset validation failed")
            
        # Original xView processing logic here...
        """
        Process xView dataset with proper class mapping, focusing only on ship-related objects.
        Uses string class names instead of numeric indices.
        """
        logging.info("Starting xView dataset processing (ships only)...")
        
        # Define ship-related classes with string names
        ship_classes = {
            23: "Maritime_Vessel",
            24: "Motorboat",
            25: "Sailboat",
            26: "Tugboat",
            27: "Barge",
            28: "Fishing_Vessel",
            29: "Ferry",
            30: "Yacht",
            31: "Container_Ship",
            32: "Oil_Tanker"
        }
        
        # xView classes mapping (11-94 to string names)
        # Only including ship-related mappings for clarity
        xview_class_mapping = {
            40: "Maritime_Vessel",  # Original index 23
            41: "Motorboat",       # Original index 24
            42: "Sailboat",        # Original index 25
            44: "Tugboat",         # Original index 26
            45: "Barge",           # Original index 27
            47: "Fishing_Vessel",  # Original index 28
            49: "Ferry",           # Original index 29
            50: "Yacht",           # Original index 30
            51: "Container_Ship",  # Original index 31
            52: "Oil_Tanker"       # Original index 32
        }
        
        # Set up paths
        train_image_dir = os.path.join(self.config.input_dir, 'train_images')
        label_path = os.path.join(self.config.input_dir, 'train_labels', 'xView_train.geojson')
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
            
        # Create output directories
        train_output_dir = os.path.join(self.config.output_dir, 'train')
        os.makedirs(os.path.join(train_output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_output_dir, 'labels'), exist_ok=True)
        if self.config.visualize:
            os.makedirs(os.path.join(train_output_dir, 'visualizations'), exist_ok=True)
        
        # Load and parse geojson
        with open(label_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Group annotations by image
        image_annotations = {}
        skipped_labels = 0
        processed_ship_count = 0
        
        # First pass: collect only images with ship annotations
        for feature in geojson_data['features']:
            try:
                image_id = feature['properties']['image_id']
                original_class_id = int(feature['properties']['type_id'])
                
                # Check if it's a ship class
                if original_class_id in xview_class_mapping:
                    class_name = xview_class_mapping[original_class_id]
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    
                    # Extract bounding box coordinates
                    bounds = feature['properties']['bounds_imcoords'].split(',')
                    xmin, ymin, xmax, ymax = map(float, bounds)
                    
                    # Store class name and absolute coordinates
                    abs_coords = [
                        xmin, ymin,  # x1,y1: top-left
                        xmax, ymin,  # x2,y2: top-right
                        xmax, ymax,  # x3,y3: bottom-right
                        xmin, ymax   # x4,y4: bottom-left
                    ]
                    image_annotations[image_id].append([class_name] + abs_coords)
                    processed_ship_count += 1
                else:
                    skipped_labels += 1
                    
            except Exception as e:
                logging.error(f"Error processing feature for image {image_id}: {str(e)}")
                skipped_labels += 1
                continue
        
        logging.info(f"Found {processed_ship_count} ship objects in {len(image_annotations)} images")
        logging.info(f"Skipped {skipped_labels} non-ship or invalid labels")
        
        # Process each image that contains ships
        processed_count = 0
        skipped_count = 0
        
        for image_id, annotations in image_annotations.items():
            try:
                image_path = os.path.join(train_image_dir, image_id)
                if not os.path.exists(image_path):
                    logging.warning(f"Image not found: {image_path}")
                    skipped_count += 1
                    continue
                
                # Load and process image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                img_width, img_height = image.size
                
                # Convert absolute coordinates to relative
                relative_labels = []
                for ann in annotations:
                    class_name = ann[0]  # Already a string
                    abs_coords = ann[1:]
                    rel_coords = convert_absolute_to_relative(abs_coords, img_width, img_height)
                    relative_labels.append([class_name] + rel_coords)
                
                # Process image (padding/tiling as needed)
                result = process_image_and_labels(
                    image,
                    relative_labels,
                    self.config.target_size[0],
                    self.config.target_size[1]
                )
                
                output_prefix = "xView"
                
                # Handle result (could be padded image or tiled images)
                if isinstance(result, tuple):
                    # Single processed image
                    processed_img, processed_labels = result
                    output_name = f"{output_prefix}_{os.path.splitext(image_id)[0]}"
                    
                    try:
                        # Save processed image
                        processed_img.save(
                            os.path.join(train_output_dir, 'images', f"{output_name}.png")
                        )
                        
                        # Save labels
                        with open(os.path.join(train_output_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                            for label in processed_labels:
                                label_str = ' '.join(map(str, label))
                                f.write(f"{label_str}\n")
                        
                        # Create visualization if enabled
                        if self.config.visualize:
                            vis_img = visualize_labels(processed_img, processed_labels)
                            vis_path = os.path.join(train_output_dir, 'visualizations', f"{output_name}_vis.png")
                            vis_img.save(vis_path)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logging.error(f"Error saving processed results for {image_id}: {str(e)}")
                        skipped_count += 1
                        continue
                        
                elif isinstance(result, list):
                    # Tiled output
                    for idx, (tile_img, tile_labels, (tx, ty)) in enumerate(result):
                        # Check if tile contains ships (now using string class names)
                        if any(label[0] in ship_classes.values() for label in tile_labels):
                            output_name = f"{output_prefix}_{os.path.splitext(image_id)[0]}_tile{tx}_{ty}"
                            
                            try:
                                # Save image tile
                                tile_img.save(os.path.join(train_output_dir, 'images', f"{output_name}.png"))
                                
                                # Save labels
                                with open(os.path.join(train_output_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                                    for label in tile_labels:
                                        label_str = ' '.join(map(str, label))
                                        f.write(f"{label_str}\n")
                                
                                # Create visualization if enabled
                                if self.config.visualize:
                                    vis_img = visualize_labels(tile_img, tile_labels)
                                    vis_path = os.path.join(train_output_dir, 'visualizations', f"{output_name}_vis.png")
                                    vis_img.save(vis_path)
                                
                                processed_count += 1
                                
                            except Exception as e:
                                logging.error(f"Error saving tile results for {image_id}: {str(e)}")
                                continue
                
                if processed_count % 100 == 0:
                    logging.info(f"Processed {processed_count} images")
                
            except Exception as e:
                logging.error(f"Error processing {image_id}: {str(e)}")
                skipped_count += 1
                continue
        
        logging.info(f"Completed processing {processed_count} xView images")
        logging.info(f"Skipped {skipped_count} images")

# ShipRSImageNet Implementation
class ShipRSImageNetValidator(DatasetValidator):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        
    def validate_structure(self) -> bool:
        voc_base = os.path.join(self.input_dir, 'VOC_Format')
        required_dirs = ['Annotations', 'ImageSets', 'JPEGImages']
        return all(os.path.exists(os.path.join(voc_base, d)) for d in required_dirs)
    
    def validate_files(self) -> Tuple[bool, Set[str]]:
        missing_files = set()
        imagesets_dir = os.path.join(self.input_dir, 'VOC_Format', 'ImageSets')
        required_files = ['train.txt', 'val.txt']
        
        for file in required_files:
            if not os.path.exists(os.path.join(imagesets_dir, file)):
                missing_files.add(file)
                
        return len(missing_files) == 0, missing_files

class ShipRSImageNetProcessor(BaseDatasetProcessor):
    def _create_validator(self) -> DatasetValidator:
        return ShipRSImageNetValidator(self.config.input_dir)
        
    def process(self) -> None:
        if not self.validate():
            raise ValueError("Dataset validation failed")
            
        # Original ShipRSImageNet processing logic here...
        """
        Process ShipRSImageNet_V1 dataset, converting annotations to DOTA format
        with relative coordinates and handling image padding/visualization.
        """
        logging.info("Starting ShipRSImageNet_V1 dataset processing...")

        # Set up paths
        voc_base = os.path.join(self.config.input_dir, 'VOC_Format')
        annotation_dir = os.path.join(voc_base, 'Annotations')
        image_dir = os.path.join(voc_base, 'JPEGImages')
        imagesets_dir = os.path.join(voc_base, 'ImageSets')

        # Verify directories exist
        for dir_path in [voc_base, annotation_dir, image_dir, imagesets_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Read train/val splits
        train_basenames = read_imageset_file(os.path.join(imagesets_dir, 'train.txt'))
        val_basenames = read_imageset_file(os.path.join(imagesets_dir, 'val.txt'))

        logging.info(f"Found {len(train_basenames)} training and {len(val_basenames)} validation samples")

        # Create output directories
        for split in ['train', 'val']:
            split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
            if self.config.visualize:
                os.makedirs(os.path.join(split_dir, 'visualizations'), exist_ok=True)

        def process_split(basenames: List[str], split: str):
            processed_count = 0
            skipped_count = 0

            logging.info(f"Starting processing of {split} split with {len(basenames)} images")

            for basename in basenames:
                try:
                    # Construct file paths
                    xml_path = os.path.join(annotation_dir, f"{basename}.xml")
                    img_path = os.path.join(image_dir, f"{basename}.bmp")

                    # Verify files exist
                    if not os.path.exists(xml_path) or not os.path.exists(img_path):
                        logging.warning(f"Missing files for {basename}")
                        skipped_count += 1
                        continue

                    # Load and validate image
                    try:
                        image = Image.open(img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                    except Exception as e:
                        logging.error(f"Failed to load image {img_path}: {str(e)}")
                        skipped_count += 1
                        continue

                    # Parse XML
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                    except ET.ParseError as e:
                        logging.error(f"Failed to parse XML {xml_path}: {str(e)}")
                        skipped_count += 1
                        continue

                    # Get image size from XML
                    size_elem = root.find('size')
                    if size_elem is None:
                        logging.warning(f"No size information in {xml_path}")
                        skipped_count += 1
                        continue

                    img_width = int(size_elem.find('width').text)
                    img_height = int(size_elem.find('height').text)

                    # Extract annotations
                    labels = []
                    for obj in root.findall('object'):
                        try:
                            # Get and process class name
                            name_elem = obj.find('name')
                            if name_elem is None:
                                continue
                            class_name = parse_shiprsimagenet_class(name_elem.text)

                            # Get polygon coordinates
                            polygon = obj.find('polygon')
                            if polygon is None:
                                continue

                            # Extract absolute coordinates
                            coords = []
                            for i in range(1, 5):
                                x = float(polygon.find(f'x{i}').text)
                                y = float(polygon.find(f'y{i}').text)
                                coords.extend([x, y])

                            # Convert to relative coordinates
                            rel_coords = convert_absolute_to_relative(coords, img_width, img_height)
                            labels.append([class_name] + rel_coords)

                        except Exception as e:
                            logging.warning(f"Error processing object in {xml_path}: {str(e)}")
                            continue

                    if not labels:
                        logging.warning(f"No valid labels found in {xml_path}")
                        skipped_count += 1
                        continue

                    # Process image and labels (padding/tiling if needed)
                    result = process_image_and_labels(
                        image,
                        labels,
                        self.config.target_size[0],
                        self.config.target_size[1]
                    )

                    # Get dataset source
                    source_elem = root.find('source/dataset_source')
                    dataset_source = source_elem.text if source_elem is not None else "unknown"

                    # Save processed results
                    if isinstance(result, tuple):
                        processed_img, processed_labels = result
                        output_name = f"ShipRSImageNet_V1_{dataset_source}_{basename}"

                        # Save processed image
                        output_img_path = os.path.join(
                            self.config.output_dir, split, 'images', f"{output_name}.png"
                        )
                        processed_img.save(output_img_path)

                        # Save labels
                        output_label_path = os.path.join(
                            self.config.output_dir, split, 'labels', f"{output_name}.txt"
                        )
                        with open(output_label_path, 'w') as f:
                            for label in processed_labels:
                                label_str = ' '.join(map(str, label))
                                f.write(f"{label_str}\n")

                        # Create visualization if enabled
                        if self.config.visualize:
                            vis_img = visualize_labels(processed_img, processed_labels)
                            vis_path = os.path.join(
                                self.config.output_dir, split, 'visualizations',
                                f"{output_name}_vis.png"
                            )
                            vis_img.save(vis_path)

                        processed_count += 1
                        if processed_count % 100 == 0:
                            logging.info(f"Processed {processed_count} images in {split} split")

                except Exception as e:
                    logging.error(f"Error processing {basename}: {str(e)}")
                    skipped_count += 1
                    continue

            return processed_count, skipped_count

        # Process train and val splits
        train_processed, train_skipped = process_split(train_basenames, 'train')
        val_processed, val_skipped = process_split(val_basenames, 'val')

        # Log final statistics
        logging.info(f"Completed ShipRSImageNet_V1 processing:")
        logging.info(f"Train: {train_processed} processed, {train_skipped} skipped")
        logging.info(f"Val: {val_processed} processed, {val_skipped} skipped")

class KARIShipDetValidator(DatasetValidator):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        
    def validate_structure(self) -> bool:
        """Validate KARI ship detection dataset directory structure"""
        required_dirs = [
            'dota/train/annfiles',
            'dota/train/images',
            'dota/val/annfiles',
            'dota/val/images'
        ]
        return all(os.path.exists(os.path.join(self.input_dir, d)) for d in required_dirs)
    
    def validate_files(self) -> Tuple[bool, Set[str]]:
        """Validate matching between annotation files and images"""
        missing_files = set()
        
        for split in ['train', 'val']:
            ann_dir = os.path.join(self.input_dir, 'dota', split, 'annfiles')
            img_dir = os.path.join(self.input_dir, 'dota', split, 'images')
            
            # Get all annotation files and image files
            ann_files = {os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith('.txt')}
            img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')}
            
            # Check for missing pairs
            for ann_base in ann_files:
                if ann_base not in img_files:
                    missing_files.add(f"{split}/images/{ann_base}.png")
                    
            for img_base in img_files:
                if img_base not in ann_files:
                    missing_files.add(f"{split}/annfiles/{img_base}.txt")
        
        return len(missing_files) == 0, missing_files

class KARIShipDetProcessor(BaseDatasetProcessor):
    def _create_validator(self) -> DatasetValidator:
        return KARIShipDetValidator(self.config.input_dir)
        
    def process(self) -> None:
        """Process KARI ship detection dataset"""
        if not self.validate():
            raise ValueError("Dataset validation failed")
            
        logging.info("Starting KARI ship detection dataset processing...")
        
        for split in ['train', 'val']:
            ann_dir = os.path.join(self.config.input_dir, 'dota', split, 'annfiles')
            img_dir = os.path.join(self.config.input_dir, 'dota', split, 'images')
            
            # Create output directories
            output_split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(os.path.join(output_split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_split_dir, 'labels'), exist_ok=True)
            if self.config.visualize:
                os.makedirs(os.path.join(output_split_dir, 'visualizations'), exist_ok=True)
            
            # Get all annotation files
            ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
            
            processed_count = 0
            skipped_count = 0
            
            for ann_file in ann_files:
                try:
                    basename = os.path.splitext(ann_file)[0]
                    img_path = os.path.join(img_dir, f"{basename}.png")
                    ann_path = os.path.join(ann_dir, ann_file)
                    
                    # Skip if image doesn't exist
                    if not os.path.exists(img_path):
                        logging.warning(f"Image not found for annotation: {ann_file}")
                        skipped_count += 1
                        continue
                    
                    # Load and process image
                    image = Image.open(img_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    img_width, img_height = image.size
                    
                    # Read and process annotations
                    labels = []
                    with open(ann_path, 'r') as f:
                        for line in f:
                            try:
                                # Parse line: 8 coordinates + class_name + difficulty
                                parts = line.strip().split()
                                if len(parts) < 10:  # At least 8 coords + class_name + difficulty
                                    continue
                                
                                # Extract coordinates and convert to relative
                                coords = [float(x) for x in parts[:8]]
                                class_name = parts[8]  # Get class name
                                # Ignore difficulty (parts[9])
                                
                                # Convert coordinates to relative
                                rel_coords = convert_absolute_to_relative(coords, img_width, img_height)
                                
                                # Format: [class_name, x1, y1, x2, y2, x3, y3, x4, y4]
                                labels.append([class_name] + rel_coords)
                                
                            except Exception as e:
                                logging.warning(f"Error processing annotation in {ann_file}: {str(e)}")
                                continue
                    
                    # Process image and labels
                    result = process_image_and_labels(
                        image,
                        labels,
                        self.config.target_size[0],
                        self.config.target_size[1]
                    )
                    
                    # Save results
                    output_prefix = "KARI"
                    if isinstance(result, tuple):
                        # Single processed image
                        processed_img, processed_labels = result
                        output_name = f"{output_prefix}_{basename}"
                        
                        # Save processed image
                        processed_img.save(
                            os.path.join(output_split_dir, 'images', f"{output_name}.png")
                        )
                        
                        # Save labels
                        with open(os.path.join(output_split_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                            for label in processed_labels:
                                label_str = ' '.join(map(str, label))
                                f.write(f"{label_str}\n")
                        
                        # Create visualization if enabled
                        if self.config.visualize:
                            vis_img = visualize_labels(processed_img, processed_labels)
                            vis_path = os.path.join(output_split_dir, 'visualizations', f"{output_name}_vis.png")
                            vis_img.save(vis_path)
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logging.info(f"Processed {processed_count} images in {split} split")
                        
                except Exception as e:
                    logging.error(f"Error processing {ann_file}: {str(e)}")
                    skipped_count += 1
                    continue
            
            logging.info(f"Completed processing {split} split:")
            logging.info(f"Processed {processed_count} images")
            logging.info(f"Skipped {skipped_count} images")

class DOTAv15Validator(DatasetValidator):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        
    def validate_structure(self) -> bool:
        """Validate DOTA v1.5 dataset directory structure"""
        required_dirs = [
            'train/images',
            'train/labelTxt-v1.5/DOTA-v1.5',
            'val/images',
            'val/labelTxt-v1.5/DOTA-v1.5',
        ]
        return all(os.path.exists(os.path.join(self.input_dir, d)) for d in required_dirs)
    
    def validate_files(self) -> Tuple[bool, Set[str]]:
        """Validate matching between annotation files and images"""
        missing_files = set()
        
        # Validate train and val splits
        for split in ['train', 'val']:
            img_dir = os.path.join(self.input_dir, split, 'images')
            ann_dir = os.path.join(self.input_dir, split, 'labelTxt-v1.5/DOTA-v1.5')
            
            img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')}
            ann_files = {os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith('.txt')}
            
            # Check for missing pairs
            for img_base in img_files:
                if img_base not in ann_files:
                    missing_files.add(f"{split}/labelTxt-v1.5/DOTA-v1.5/{img_base}.txt")
            for ann_base in ann_files:
                if ann_base not in img_files:
                    missing_files.add(f"{split}/images/{ann_base}.png")
        
        return len(missing_files) == 0, missing_files

class DOTAv15Processor(BaseDatasetProcessor):
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.ship_classes = {'ship'}  # Only ship class
    
    def _create_validator(self) -> DatasetValidator:
        return DOTAv15Validator(self.config.input_dir)
    
    def _process_metadata(self, lines: List[str]) -> int:
        """Process metadata lines and return the number of metadata lines"""
        for i, line in enumerate(lines):
            if all(char.isdigit() or char in {'.', ' ', '\t', '-'} for char in line.strip()):
                return i
        return 0
    
    def _has_ship_related_objects(self, labels: List[str]) -> bool:
        """Check if any label contains ship-related classes"""
        for label in labels:
            parts = label.strip().split()
            if len(parts) >= 9:  # 8 coordinates + class name
                class_name = parts[8].lower()
                if class_name in self.ship_classes:
                    return True
        return False
    
    def process(self) -> None:
        """Process DOTA v1.5 dataset"""
        if not self.validate():
            raise ValueError("Dataset validation failed")
            
        logging.info("Starting DOTA v1.5 dataset processing...")
        
        # Process train and val splits
        for split in ['train', 'val']:
            img_dir = os.path.join(self.config.input_dir, split, 'images')
            ann_dir = os.path.join(self.config.input_dir, split, 'labelTxt-v1.5/DOTA-v1.5')
            
            # Create output directories
            output_split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(os.path.join(output_split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_split_dir, 'labels'), exist_ok=True)
            if self.config.visualize:
                os.makedirs(os.path.join(output_split_dir, 'visualizations'), exist_ok=True)
            
            # Get all annotation files
            ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
            
            processed_count = 0
            skipped_count = 0
            ship_related_count = 0
            
            for ann_file in ann_files:
                try:
                    basename = os.path.splitext(ann_file)[0]
                    img_path = os.path.join(img_dir, f"{basename}.png")
                    ann_path = os.path.join(ann_dir, ann_file)
                    
                    # Skip if image doesn't exist
                    if not os.path.exists(img_path):
                        logging.warning(f"Image not found for annotation: {ann_file}")
                        skipped_count += 1
                        continue
                    
                    # Read annotation file
                    with open(ann_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Skip metadata lines
                    metadata_lines = self._process_metadata(lines)
                    annotation_lines = lines[metadata_lines:]
                    
                    # Skip if no ship-related objects
                    if not self._has_ship_related_objects(annotation_lines):
                        skipped_count += 1
                        continue
                    
                    ship_related_count += 1
                    
                    # Load and process image
                    image = Image.open(img_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    img_width, img_height = image.size
                    
                    # Process annotations
                    labels = []
                    for line in annotation_lines:
                        try:
                            parts = line.strip().split()
                            if len(parts) < 10:  # 8 coords + class_name + difficulty
                                continue
                            
                            # Extract coordinates and convert to float
                            coords = [float(x) for x in parts[:8]]
                            class_name = parts[8]
                            # Ignore difficulty (parts[9])
                            
                            # Convert to relative coordinates
                            rel_coords = convert_absolute_to_relative(coords, img_width, img_height)
                            
                            # Only append ship class objects
                            if class_name.lower() == 'ship':
                                # Format: [class_name, x1, y1, x2, y2, x3, y3, x4, y4]
                                labels.append([class_name] + rel_coords)
                            
                        except Exception as e:
                            logging.warning(f"Error processing annotation in {ann_file}: {str(e)}")
                            continue
                    
                    # Process image and labels
                    result = process_image_and_labels(
                        image,
                        labels,
                        self.config.target_size[0],
                        self.config.target_size[1]
                    )
                    
                    # Save results
                    output_prefix = "DOTA-v1.5"
                    if isinstance(result, tuple):
                        # Single processed image (padded case)
                        processed_img, processed_labels = result
                        output_name = f"{output_prefix}_{basename}"
                        
                        # Save processed image
                        processed_img.save(
                            os.path.join(output_split_dir, 'images', f"{output_name}.png")
                        )
                        
                        # Save labels
                        with open(os.path.join(output_split_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                            for label in processed_labels:
                                label_str = ' '.join(map(str, label))
                                f.write(f"{label_str}\n")
                        
                        # Create visualization if enabled
                        if self.config.visualize:
                            vis_img = visualize_labels(processed_img, processed_labels)
                            vis_path = os.path.join(output_split_dir, 'visualizations', f"{output_name}_vis.png")
                            vis_img.save(vis_path)
                            
                    elif isinstance(result, list):
                        # Multiple tiles case
                        for idx, (tile_img, tile_labels, (tx, ty)) in enumerate(result):
                            # Only save tiles that contain objects
                            if tile_labels:
                                output_name = f"{output_prefix}_{basename}_tile{tx}_{ty}"
                                
                                # Save image tile
                                tile_img.save(
                                    os.path.join(output_split_dir, 'images', f"{output_name}.png")
                                )
                                
                                # Save labels
                                with open(os.path.join(output_split_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                                    for label in tile_labels:
                                        label_str = ' '.join(map(str, label))
                                        f.write(f"{label_str}\n")
                                
                                # Create visualization if enabled
                                if self.config.visualize:
                                    vis_img = visualize_labels(tile_img, tile_labels)
                                    vis_path = os.path.join(output_split_dir, 'visualizations', f"{output_name}_vis.png")
                                    vis_img.save(vis_path)
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logging.info(f"Processed {processed_count} images in {split} split")
                        
                except Exception as e:
                    logging.error(f"Error processing {ann_file}: {str(e)}")
                    skipped_count += 1
                    continue
            
            logging.info(f"Completed processing {split} split:")
            logging.info(f"Found {ship_related_count} images with ship-related objects")
            logging.info(f"Successfully processed {processed_count} images")
            logging.info(f"Skipped {skipped_count} images")

class AirbusShipDetValidator(DatasetValidator):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        
    def validate_structure(self) -> bool:
        """Validate Airbus ship detection dataset directory structure"""
        required_dirs = ['train_v2', 'test_v2']
        required_files = ['train_ship_segmentations_v2.csv']
        
        # Check required directories
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(self.input_dir, dir_name)):
                logging.error(f"Missing required directory: {dir_name}")
                return False
                
        # Check required files
        for file_name in required_files:
            if not os.path.exists(os.path.join(self.input_dir, file_name)):
                logging.error(f"Missing required file: {file_name}")
                return False
                
        return True
    
    def validate_files(self) -> Tuple[bool, Set[str]]:
        """Validate matching between images and annotations"""
        missing_files = set()
        
        # Read CSV file
        csv_path = os.path.join(self.input_dir, 'train_ship_segmentations_v2.csv')
        try:
            df = pd.read_csv(csv_path)
            image_ids = df['ImageId'].unique()
            
            # Check if all images in CSV exist in train_v2 directory
            train_dir = os.path.join(self.input_dir, 'train_v2')
            for img_id in image_ids:
                img_path = os.path.join(train_dir, img_id)
                if not os.path.exists(img_path):
                    missing_files.add(f"train_v2/{img_id}")
                    
        except Exception as e:
            logging.error(f"Error validating files: {str(e)}")
            return False, missing_files
            
        return len(missing_files) == 0, missing_files

class AirbusShipDetProcessor(BaseDatasetProcessor):
    def _create_validator(self) -> DatasetValidator:
        return AirbusShipDetValidator(self.config.input_dir)
        
    def _rle_decode(self, mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode RLE (Run-Length Encoding) mask with proper transposition
        
        Args:
            mask_rle: Run-length encoded mask string
            shape: Image shape (height, width)
            
        Returns:
            Binary mask as numpy array
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1  # Adjust for 1-based indexing
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # Important: Transpose the mask to match the correct orientation
        return img.reshape(shape).T
    
    def _get_oriented_bbox(self, mask: np.ndarray) -> Optional[List[float]]:
        """
        Convert binary mask to oriented bounding box coordinates with proper ordering
        
        Args:
            mask: Binary mask as numpy array
            
        Returns:
            List of coordinates [x1,y1,x2,y2,x3,y3,x4,y4] in clockwise order
            starting from top-left, or None if no valid contour found
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Process each contour and return the largest one
        largest_box = None
        largest_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Sort points based on their y-coordinates
                box = box[np.argsort(box[:, 1])]
                
                # Determine top-left and top-right based on x-coordinates
                top_two = box[:2]
                if top_two[0][0] > top_two[1][0]:
                    top_left, top_right = top_two[1], top_two[0]
                else:
                    top_left, top_right = top_two[0], top_two[1]
                
                # Determine bottom-left and bottom-right based on x-coordinates
                bottom_two = box[2:]
                if bottom_two[0][0] > bottom_two[1][0]:
                    bottom_left, bottom_right = bottom_two[1], bottom_two[0]
                else:
                    bottom_left, bottom_right = bottom_two[0], bottom_two[1]
                
                # Store in clockwise order: top-left, top-right, bottom-right, bottom-left
                largest_box = np.array([top_left, top_right, bottom_right, bottom_left]).flatten()
                largest_area = area
        
        return largest_box.tolist() if largest_box is not None else None
    
    def _process_single_image(self, 
                            image_path: str, 
                            masks_rle: List[str]) -> Tuple[Image.Image, List[List[float]]]:
        """
        Process single image and its RLE masks
        
        Args:
            image_path: Path to image file
            masks_rle: List of RLE encoded mask strings
            
        Returns:
            Tuple of (image, labels)
        """
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_width, img_height = image.size
        
        # Process each mask
        labels = []
        for mask_rle in masks_rle:
            if isinstance(mask_rle, str) and len(mask_rle.strip()) > 0:
                # Decode RLE mask
                mask = self._rle_decode(mask_rle, (img_height, img_width))
                
                # Convert mask to bbox
                bbox = self._get_oriented_bbox(mask)
                if bbox:
                    # Convert to relative coordinates
                    rel_coords = convert_absolute_to_relative(bbox, img_width, img_height)
                    # Add class name (just "ship" for this dataset)
                    labels.append(["ship"] + rel_coords)
        
        return image, labels
    
    def process(self) -> None:
        """Process Airbus Ship Detection Challenge dataset"""
        if not self.validate():
            raise ValueError("Dataset validation failed")
            
        logging.info("Starting Airbus Ship Detection Challenge dataset processing...")
        
        # Read CSV file
        csv_path = os.path.join(self.config.input_dir, 'train_ship_segmentations_v2.csv')
        df = pd.read_csv(csv_path)
        
        # Group by ImageId to handle multiple ships per image
        df_grouped = df.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
        
        # Filter out empty masks and convert to dictionary
        image_data = {row['ImageId']: [mask for mask in row['EncodedPixels'] if isinstance(mask, str)]
                     for _, row in df_grouped.iterrows()}
        
        # Setup output directories
        train_dir = os.path.join(self.config.output_dir, 'train')
        os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
        if self.config.visualize:
            os.makedirs(os.path.join(train_dir, 'visualizations'), exist_ok=True)
            
        processed_count = 0
        skipped_count = 0
        
        # Process each image
        for image_id, masks_rle in image_data.items():
            try:
                # Skip images with less than 2 ships
                if len(masks_rle) < 2:  # Add this filtering
                    skipped_count += 1
                    logging.info(f"Skipping {image_id}: Less than 2 ships")
                    continue
                
                image_path = os.path.join(self.config.input_dir, 'train_v2', image_id)
                
                # Skip if image doesn't exist
                if not os.path.exists(image_path):
                    logging.warning(f"Image not found: {image_id}")
                    skipped_count += 1
                    continue
                    
                # Process image and its masks
                image, labels = self._process_single_image(image_path, masks_rle)
                
                # Skip if no valid ships detected
                if not labels:
                    skipped_count += 1
                    continue
                    
                # Process image and labels (padding/tiling if needed)
                result = process_image_and_labels(
                    image,
                    labels,
                    self.config.target_size[0],
                    self.config.target_size[1]
                )
                
                # Save results
                output_prefix = "airbus-ship-det-challenge"
                if isinstance(result, tuple):
                    # Single processed image
                    processed_img, processed_labels = result
                    output_name = f"{output_prefix}_{os.path.splitext(image_id)[0]}"
                    
                    # Save processed image
                    processed_img.save(
                        os.path.join(train_dir, 'images', f"{output_name}.png")
                    )
                    
                    # Save labels
                    with open(os.path.join(train_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                        for label in processed_labels:
                            label_str = ' '.join(map(str, label))
                            f.write(f"{label_str}\n")
                    
                    # Create visualization if enabled
                    if self.config.visualize:
                        vis_img = visualize_labels(processed_img, processed_labels)
                        vis_path = os.path.join(train_dir, 'visualizations', f"{output_name}_vis.png")
                        vis_img.save(vis_path)
                        
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logging.info(f"Processed {processed_count} images")
                        
            except Exception as e:
                logging.error(f"Error processing {image_id}: {str(e)}")
                skipped_count += 1
                continue
                
        logging.info(f"Completed processing:")
        logging.info(f"Successfully processed {processed_count} images")
        logging.info(f"Skipped {skipped_count} images")

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Set
import logging
from pathlib import Path
import cv2

class FAIR1MShipValidator(DatasetValidator):
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.ship_classes = {
            "Passenger Ship", "Motorboat", "Fishing Boat", 
            "Tugboat", "other-ship", "Engineering Ship", 
            "Liquid Cargo Ship", "Dry Cargo Ship", "Warship"
        }
        
    def validate_structure(self) -> bool:
        """Validate FAIR1M dataset directory structure"""
        base_dir = os.path.join(self.input_dir, 'ver2')  # Add ver2 as base directory
        
        required_dirs = [
            'test/images',
            'train/part1/images',
            'train/part1/labelXml',
            'validation/images',
            'validation/labelXml'
        ]
        
        for dir_path in required_dirs:
            full_path = os.path.join(base_dir, dir_path)  # Changed from self.input_dir
            if not os.path.exists(full_path):
                logging.error(f"Missing required directory: {dir_path}")
                return False
        return True
    
    def validate_files(self) -> Tuple[bool, Set[str]]:
        """Validate matching between images and annotations"""
        missing_files = set()
        
        # Check train and validation sets
        for split in ['train/part1', 'validation']:
            img_dir = os.path.join(self.input_dir, 'ver2', split, 'images')
            xml_dir = os.path.join(self.input_dir, 'ver2', split, 'labelXml')
            
            if os.path.exists(img_dir) and os.path.exists(xml_dir):
                # Get all image and XML files
                img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.tif')}
                xml_files = {os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith('.xml')}
                
                # Check for missing pairs
                for img_id in img_files:
                    if img_id not in xml_files:
                        missing_files.add(f"{split}/labelXml/{img_id}.xml")
                        
                for xml_id in xml_files:
                    if xml_id not in img_files:
                        missing_files.add(f"{split}/images/{xml_id}.tif")
                
        return len(missing_files) == 0, missing_files

class FAIR1MShipProcessor(BaseDatasetProcessor):
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.ship_classes = {
            "Passenger Ship", "Motorboat", "Fishing Boat", 
            "Tugboat", "other-ship", "Engineering Ship", 
            "Liquid Cargo Ship", "Dry Cargo Ship", "Warship"
        }
    
    def _create_validator(self) -> DatasetValidator:
        return FAIR1MShipValidator(self.config.input_dir)
    
    def _parse_points(self, points_text: list) -> np.ndarray:
        """Convert point strings to numpy array of coordinates"""
        points = []
        for point in points_text:
            x, y = map(float, point.text.split(','))
            points.append([x, y])
        points = np.array(points)
        
        # Convert 5-point OBB (where first and last points are same) 
        # to 4-point OBB in clockwise order
        return points[:-1]  # Remove the last repeated point
    
    def _process_single_xml(self, xml_path: str) -> List[List[float]]:
        """
        Process single XML annotation file
        Returns list of [class_name, x1, y1, x2, y2, x3, y3, x4, y4]
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        labels = []
        objects = root.findall('./objects/object')
        
        for obj in objects:
            # Check if object is a ship class
            name = obj.find('./possibleresult/name').text
            if name not in self.ship_classes:
                continue
                
            # Get points
            points = self._parse_points(obj.findall('./points/point'))
            if len(points) != 4:
                logging.warning(f"Skipping object with {len(points)} points in {xml_path}")
                continue
                
            # Flatten points to [x1,y1,x2,y2,x3,y3,x4,y4] format
            coords = points.flatten().tolist()
            labels.append([name] + coords)
            
        return labels
    
    def _has_ship_objects(self, xml_path: str) -> bool:
        """Check if XML file contains any ship objects"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('./objects/object'):
            name = obj.find('./possibleresult/name').text
            if name in self.ship_classes:
                return True
        return False
    
    def process(self) -> None:
        """Process FAIR1M ship dataset"""
        if not self.validate():
            raise ValueError("Dataset validation failed")
            
        logging.info("Starting FAIR1M ship dataset processing...")
        base_dir = os.path.join(self.config.input_dir, 'ver2')
        
        # Process train and validation splits
        for split in ['train/part1', 'validation']:
            img_dir = os.path.join(base_dir, split, 'images')
            xml_dir = os.path.join(base_dir, split, 'labelXml')
            
            # Setup output directories
            output_split = 'train' if 'train' in split else 'val'
            output_dir = os.path.join(self.config.output_dir, output_split)
            os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
            if self.config.visualize:
                os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
                
            processed_count = 0
            skipped_count = 0
            
            # Process each XML file
            for xml_file in os.listdir(xml_dir):
                try:
                    if not xml_file.endswith('.xml'):
                        continue
                        
                    xml_path = os.path.join(xml_dir, xml_file)
                    
                    # Skip if no ship objects
                    if not self._has_ship_objects(xml_path):
                        skipped_count += 1
                        continue
                        
                    # Get corresponding image
                    img_file = os.path.splitext(xml_file)[0] + '.tif'
                    img_path = os.path.join(img_dir, img_file)
                    
                    if not os.path.exists(img_path):
                        logging.warning(f"Image not found: {img_file}")
                        skipped_count += 1
                        continue
                        
                    # Load and process image
                    image = Image.open(img_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    # Get labels
                    labels = self._process_single_xml(xml_path)
                    if not labels:
                        skipped_count += 1
                        continue
                        
                    # Convert to relative coordinates
                    img_width, img_height = image.size
                    rel_labels = []
                    for label in labels:
                        class_name = label[0]
                        coords = label[1:]
                        rel_coords = convert_absolute_to_relative(coords, img_width, img_height)
                        rel_labels.append([class_name] + rel_coords)
                        
                    # Process image and labels
                    result = process_image_and_labels(
                        image,
                        rel_labels,
                        self.config.target_size[0],
                        self.config.target_size[1]
                    )
                    
                    # Save results
                    output_name = f"FAIR1M_{os.path.splitext(img_file)[0]}"
                    if isinstance(result, tuple):
                        processed_img, processed_labels = result
                        
                        # Save processed image
                        processed_img.save(
                            os.path.join(output_dir, 'images', f"{output_name}.png")
                        )
                        
                        # Save labels
                        with open(os.path.join(output_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                            for label in processed_labels:
                                label_str = ' '.join(map(str, label))
                                f.write(f"{label_str}\n")
                        
                        # Create visualization if enabled
                        if self.config.visualize:
                            vis_img = visualize_labels(processed_img, processed_labels)
                            vis_path = os.path.join(output_dir, 'visualizations', f"{output_name}_vis.png")
                            vis_img.save(vis_path)
                            
                        processed_count += 1
                        if processed_count % 100 == 0:
                            logging.info(f"Processed {processed_count} images in {split}")
                            
                except Exception as e:
                    logging.error(f"Error processing {xml_file}: {str(e)}")
                    skipped_count += 1
                    continue
                    
            logging.info(f"Completed processing {split}:")
            logging.info(f"Successfully processed {processed_count} images")
            logging.info(f"Skipped {skipped_count} images")

# Factory for creating processors
class ProcessorFactory:
    _processors = {
        'xview': XViewProcessor,
        'shiprsimagenet': ShipRSImageNetProcessor,
        'kari': KARIShipDetProcessor,
        'dotav15': DOTAv15Processor,
        'airbus': AirbusShipDetProcessor,
        'fair1m': FAIR1MShipProcessor
    }
    
    @classmethod
    def create_processor(cls, dataset_type: str, config: ProcessingConfig) -> BaseDatasetProcessor:
        processor_class = cls._processors.get(dataset_type.lower())
        if not processor_class:
            raise ValueError(f"No processor found for dataset type: {dataset_type}")
        return processor_class(config)

# Repository for storing processing results
class ProcessingRepository:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        
    def save_results(self, split: str, results: Dict[str, Any]) -> None:
        """Save processing results to specified directory"""
        pass
    
    def get_results(self, split: str) -> Dict[str, Any]:
        """Retrieve processing results"""
        pass
