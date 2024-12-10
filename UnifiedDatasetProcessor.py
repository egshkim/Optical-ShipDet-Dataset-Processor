import os
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import Enum
from PIL import Image
from utils import *
from pathlib import Path
import numpy as np
        
class CoordinateFormat(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

@dataclass
class ProcessingConfig:
    input_dir: str          # Input dataset directory
    output_dir: str         # Output directory for processed files
    target_size: Tuple[int, int]  # Target image dimensions (width, height)
    coord_format: CoordinateFormat  # RELATIVE or ABSOLUTE
    visualize: bool = False  # Enable visualization of annotations
    visualization_dir: Optional[str] = None  # Custom directory for visualizations

    def validate(self) -> None:
        """Validates configuration parameters"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        if self.target_size[0] <= 0 or self.target_size[1] <= 0:
            raise ValueError("Target dimensions must be positive")

class DatasetProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.config.validate()
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Creates necessary output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def process_xview(self) -> None:
        """
        Process xView dataset with proper class mapping, focusing only on ship-related objects.
        Only saves images and labels that contain at least one ship object.
                
        Expected directory structure:
        input_dir/
        ├── train_images/
        ├── train_labels/
        │   └── xView_train.geojson
        └── val_images/
        
        # Classes
        names:
        0: Fixed-wing Aircraft
        1: Small Aircraft
        2: Cargo Plane
        3: Helicopter
        4: Passenger Vehicle
        5: Small Car
        6: Bus
        7: Pickup Truck
        8: Utility Truck
        9: Truck
        10: Cargo Truck
        11: Truck w/Box
        12: Truck Tractor
        13: Trailer
        14: Truck w/Flatbed
        15: Truck w/Liquid
        16: Crane Truck
        17: Railway Vehicle
        18: Passenger Car
        19: Cargo Car
        20: Flat Car
        21: Tank car
        22: Locomotive
        23: Maritime Vessel
        24: Motorboat
        25: Sailboat
        26: Tugboat
        27: Barge
        28: Fishing Vessel
        29: Ferry
        30: Yacht
        31: Container Ship
        32: Oil Tanker
        33: Engineering Vehicle
        34: Tower crane
        35: Container Crane
        36: Reach Stacker
        37: Straddle Carrier
        38: Mobile Crane
        39: Dump Truck
        40: Haul Truck
        41: Scraper/Tractor
        42: Front loader/Bulldozer
        43: Excavator
        44: Cement Mixer
        45: Ground Grader
        46: Hut/Tent
        47: Shed
        48: Building
        49: Aircraft Hangar
        50: Damaged Building
        51: Facility
        52: Construction Site
        53: Vehicle Lot
        54: Helipad
        55: Storage Tank
        56: Shipping container lot
        57: Shipping Container
        58: Pylon
        59: Tower
        """
        
        logging.info("Starting xView dataset processing (ships only)...")
        
        # Define ship-related classes
        ship_class_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # Maritime vessels and variants
        
        # xView classes mapping from 11-94 to 0-59
        xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0-10
                            0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10,  # 11-24
                            11, 12, 13, 14, 15, -1, -1, 16, 17, 18, 19,    # 25-35
                            20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1,    # 36-46
                            28, -1, 29, 30, 31, 32, 33, 34, 35, 36, 37,    # 47-57
                            -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1,    # 58-68
                            -1, -1, 46, 47, 48, 49, -1, 50, 51, -1, 52,    # 69-79
                            -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1,    # 80-90
                            57, -1, 58, 59]                                 # 91-94
        
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
                
                # Map original class ID to new index and check if it's a ship
                if 11 <= original_class_id <= 94:
                    mapped_class_id = xview_class2index[original_class_id]
                    if mapped_class_id in ship_class_indices:  # Only process if it's a ship
                        if image_id not in image_annotations:
                            image_annotations[image_id] = []
                        
                        # Extract bounding box coordinates
                        bounds = feature['properties']['bounds_imcoords'].split(',')
                        xmin, ymin, xmax, ymax = map(float, bounds)
                        
                        # Convert to DOTA format
                        dota_format = [
                            mapped_class_id,  # Add mapped class_id as first element
                            xmin, ymin,  # x1,y1: top-left
                            xmax, ymin,  # x2,y2: top-right
                            xmax, ymax,  # x3,y3: bottom-right
                            xmin, ymax   # x4,y4: bottom-left
                        ]
                        image_annotations[image_id].append(dota_format)
                        processed_ship_count += 1
                    else:
                        skipped_labels += 1
                else:
                    skipped_labels += 1
                    
            except Exception as e:
                logging.error(f"Error processing feature for image {image_id}: {str(e)}")
                skipped_labels += 1
                continue
        
        logging.info(f"Found {processed_ship_count} ship objects in {len(image_annotations)} images")
        logging.info(f"Skipped {skipped_labels} non-ship or invalid labels")
        
        # Initialize visualization parameters if enabled
        vis_probability = 0.05  # Visualize roughly 5% of processed images
        if self.config.visualize:
            vis_dir = os.path.join(train_output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            random.seed(42)  # Set seed for reproducible visualization selection
        
        # Process each image that contains ships
        processed_count = 0
        
        for image_id, annotations in image_annotations.items():
            try:
                image_path = os.path.join(train_image_dir, image_id)
                if not os.path.exists(image_path):
                    logging.warning(f"Image not found: {image_path}")
                    continue
                
                # Load and process image
                image = Image.open(image_path)
                img_width, img_height = image.size
                
                # Convert absolute coordinates to relative
                relative_labels = []
                for ann in annotations:
                    class_id = ann[0]  # Already mapped class ID
                    coords = ann[1:]  # Get the 8 coordinates
                    rel_coords = convert_absolute_to_relative(coords, img_width, img_height)
                    relative_labels.append([class_id] + rel_coords)
                
                # Process image (tile or pad as needed)
                result = process_image_and_labels(
                    image,
                    relative_labels,
                    self.config.target_size[0],
                    self.config.target_size[1]
                )
                
                # Handle tiled or padded results
                if isinstance(result, list):  # Tiled output
                    for idx, (tile_img, tile_labels, (tx, ty)) in enumerate(result):
                        # Only process tiles with ship labels
                        if any(label[0] in ship_class_indices for label in tile_labels):
                            output_name = f"xView_{os.path.splitext(image_id)[0]}_tile{tx}_{ty}"
                            
                            # Save image tile
                            tile_img.save(os.path.join(train_output_dir, 'images', f"{output_name}.png"))
                            
                            # Save labels
                            with open(os.path.join(train_output_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                                for label in tile_labels:
                                    if label[0] in ship_class_indices:  # Only save ship labels
                                        label_str = ' '.join(map(str, label))
                                        f.write(label_str + '\n')
                            
                            # Visualize tile with probability
                            if self.config.visualize and random.random() < vis_probability:
                                vis_img = visualize_labels(tile_img, tile_labels)
                                vis_img.save(os.path.join(vis_dir, f"{output_name}_vis.png"))
                                logging.info(f"Visualized tile: {output_name}")
                        
                else:  # Padded output
                    processed_img, processed_labels = result
                    
                    # Only save if there are ship labels
                    if any(label[0] in ship_class_indices for label in processed_labels):
                        output_name = f"xView_{os.path.splitext(image_id)[0]}"
                        
                        # Save processed image
                        processed_img.save(os.path.join(train_output_dir, 'images', f"{output_name}.png"))
                        
                        # Save labels (only ship labels)
                        with open(os.path.join(train_output_dir, 'labels', f"{output_name}.txt"), 'w') as f:
                            for label in processed_labels:
                                if label[0] in ship_class_indices:
                                    label_str = ' '.join(map(str, label))
                                    f.write(label_str + '\n')
                        
                        # Visualize with probability
                        if self.config.visualize and random.random() < vis_probability:
                            vis_img = visualize_labels(processed_img, processed_labels)
                            vis_img.save(os.path.join(vis_dir, f"{output_name}_vis.png"))
                            logging.info(f"Visualized image: {output_name}")
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logging.info(f"Processed {processed_count} images containing ships")
                    
            except Exception as e:
                logging.error(f"Error processing {image_id}: {str(e)}")
                continue
        
        logging.info(f"Completed processing {processed_count} xView images containing ships")