from typing import List, Tuple, Union, Optional, Dict, Callable
import warnings
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import logging
import math
import os 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants for visualization
DEFAULT_BOX_COLOR = (255, 0, 0)  # Red
DEFAULT_TEXT_COLOR = (255, 255, 255)  # White
DEFAULT_LINE_THICKNESS = 2
DEFAULT_FONT_SIZE = 12
MIN_LINE_THICKNESS = 1
MAX_LINE_THICKNESS = 10
RANDOM_SEED = 42

def validate_rgb_color(color: Tuple[int, int, int]) -> bool:
    """
    Validates RGB color tuple.
    
    Args:
        color: Tuple of (R,G,B) values.
    
    Returns:
        bool: True if valid RGB color, False otherwise.
    """
    if not isinstance(color, tuple) or len(color) != 3:
        return False
    return all(isinstance(v, int) and 0 <= v <= 255 for v in color)

def validate_coordinates(coords: List[float]) -> bool:
    """
    Validates coordinate list format.
    
    Args:
        coords: List of relative coordinates.
    
    Returns:
        bool: True if valid coordinates, False otherwise.
    """
    if not isinstance(coords, list):
        return False
    if len(coords) % 2 != 0:  # Must have pairs of coordinates
        return False
    return all(isinstance(v, (int, float)) for v in coords)

def detect_coordinate_format(
    coords: List[float], 
    img_width: int, 
    img_height: int
) -> str:
    """
    Detects whether coordinates are in relative (0-1) or absolute format.
    
    Args:
        coords: List of coordinates
        img_width: Image width
        img_height: Image height
    
    Returns:
        'relative' if coordinates are between 0-1
        'absolute' if coordinates appear to be pixel values
    """
    for i, coord in enumerate(coords):
        if i % 2 == 0:  # x coordinate
            if coord > 1.0 and coord <= img_width:
                return 'absolute'
        else:  # y coordinate
            if coord > 1.0 and coord <= img_height:
                return 'absolute'
    return 'relative'

def convert_relative_to_absolute(
    coords: List[float], 
    img_width: int, 
    img_height: int
) -> List[int]:
    """
    Converts relative coordinates (0-1) to absolute pixel coordinates.
    
    Args:
        coords: List of relative coordinates (x1, y1, x2, y2, ...) between 0 and 1.
        img_width: Image width in pixels for scaling.
        img_height: Image height in pixels for scaling.
    
    Returns:
        List of absolute pixel coordinates, clamped to image boundaries if necessary.
    
    Raises:
        TypeError: If image dimensions are not integers.
        ValueError: If image dimensions are not positive.
    
    Examples:
        >>> coords = [0.1, 0.2, 0.3, 0.4]  # [x1, y1, x2, y2]
        >>> img_width, img_height = 100, 200
        >>> convert_relative_to_absolute(coords, img_width, img_height)
        [10, 40, 30, 80]  # Converted to pixel coordinates
        
        >>> # Handling out-of-bounds coordinates
        >>> coords = [-0.1, 1.2, 0.3, 0.4]
        >>> convert_relative_to_absolute(coords, img_width, img_height)
        [0, 200, 30, 80]  # Clamped to image boundaries
    """
    # Validate inputs
    if not validate_coordinates(coords):
        raise ValueError("Invalid coordinate format")
    
    if not all(0 <= coord <= 1 for coord in coords):
        warnings.warn("Some relative coordinates are out of bounds (not between 0 and 1).")
    
    if not (isinstance(img_width, int) and isinstance(img_height, int)):
        raise TypeError("Image dimensions must be integers")
    
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # Adjust out-of-bound relative coordinates and issue warnings
    for i in range(len(coords)):
        if coords[i] < 0:
            warnings.warn(f"Coordinate {coords[i]} is less than 0. Clamping to 0.")
            coords[i] = 0
        elif coords[i] > 1:
            warnings.warn(f"Coordinate {coords[i]} is greater than 1. Clamping to 1.")
            coords[i] = 1

    # Convert relative coordinates to absolute coordinates
    absolute_coords = [
        round(coords[i] * (img_width if i % 2 == 0 else img_height))
        for i in range(len(coords))
    ]

    # Ensure absolute coordinates fit within image boundaries (clamp to min/max)
    for i in range(len(absolute_coords)):
        if i % 2 == 0:  # x-coordinate
            if absolute_coords[i] < 0:
                warnings.warn(f"x-coordinate {absolute_coords[i]} is less than 0. Clamping to 0.")
                absolute_coords[i] = 0
            elif absolute_coords[i] > img_width:
                warnings.warn(f"x-coordinate {absolute_coords[i]} exceeds image width. Clamping to {img_width}.")
                absolute_coords[i] = img_width
        else:  # y-coordinate
            if absolute_coords[i] < 0:
                warnings.warn(f"y-coordinate {absolute_coords[i]} is less than 0. Clamping to 0.")
                absolute_coords[i] = 0
            elif absolute_coords[i] > img_height:
                warnings.warn(f"y-coordinate {absolute_coords[i]} exceeds image height. Clamping to {img_height}.")
                absolute_coords[i] = img_height

    return absolute_coords


def convert_absolute_to_relative(
    coords: List[int], 
    img_width: int, 
    img_height: int
) -> List[float]:
    """
    Converts absolute pixel coordinates to relative coordinates (0-1).
    
    Args:
        coords: List of absolute pixel coordinates (x1, y1, x2, y2, ...).
        img_width: Image width in pixels.
        img_height: Image height in pixels.
    
    Returns:
        List of relative coordinates between 0 and 1, rounded to 6 decimal places.
    
    Raises:
        TypeError: If image dimensions are not integers.
        ValueError: If image dimensions are not positive.
    
    Examples:
        >>> coords = [10, 40, 30, 80]  # [x1, y1, x2, y2] in pixels
        >>> img_width, img_height = 100, 200
        >>> convert_absolute_to_relative(coords, img_width, img_height)
        [0.1, 0.2, 0.3, 0.4]  # Converted to relative coordinates
        
        >>> # Handling out-of-bounds coordinates
        >>> coords = [-10, 250, 30, 80]
        >>> convert_absolute_to_relative(coords, img_width, img_height)
        [0.0, 1.0, 0.3, 0.4]  # Clamped to valid range
    """
    # Validate input
    if not (isinstance(img_width, int) and isinstance(img_height, int)):
        raise TypeError("Image dimensions must be integers")
    
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")

    # Adjust out-of-bound coordinates and issue warnings
    for i in range(len(coords)):
        if i % 2 == 0:  # x-coordinate (even index)
            if coords[i] < 0:
                warnings.warn(f"x-coordinate {coords[i]} is out of bounds (less than 0). Clamping to 0.")
                coords[i] = 0
            elif coords[i] > img_width:
                warnings.warn(f"x-coordinate {coords[i]} is out of bounds (greater than image width). Clamping to {img_width}.")
                coords[i] = img_width
        else:  # y-coordinate (odd index)
            if coords[i] < 0:
                warnings.warn(f"y-coordinate {coords[i]} is out of bounds (less than 0). Clamping to 0.")
                coords[i] = 0
            elif coords[i] > img_height:
                warnings.warn(f"y-coordinate {coords[i]} is out of bounds (greater than image height). Clamping to {img_height}.")
                coords[i] = img_height
    
    # Convert absolute coordinates to relative coordinates
    return [
        round(coords[i] / (img_width if i % 2 == 0 else img_height), 6)
        for i in range(len(coords))
    ]


def convert_yolo_to_dota(yolo_bbox: List[float]) -> List[float]:
    """
    Converts YOLO format bounding box to DOTA format coordinates.
    
    Args:
        yolo_bbox: YOLO format bounding box [x_center, y_center, width, height]
                  where all values are relative (0-1).
    
    Returns:
        List of 8 coordinates in DOTA format [x1, y1, x2, y2, x3, y3, x4, y4]
        representing the four corners in clockwise order starting from top-left.
    
    Examples:
        >>> yolo_bbox = [0.5, 0.5, 0.2, 0.3]  # center_x, center_y, width, height
        >>> convert_yolo_to_dota(yolo_bbox)
        [0.4, 0.35, 0.6, 0.35, 0.6, 0.65, 0.4, 0.65]  # Four corners clockwise
    """
    center_x, center_y, width, height = yolo_bbox
    
    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2
    
    # Calculate corners (clockwise from top-left)
    x1, y1 = center_x - half_width, center_y - half_height  # top-left
    x2, y2 = center_x + half_width, center_y - half_height  # top-right
    x3, y3 = center_x + half_width, center_y + half_height  # bottom-right
    x4, y4 = center_x - half_width, center_y + half_height  # bottom-left
    
    return [x1, y1, x2, y2, x3, y3, x4, y4]

def convert_coco_to_dota(coco_bbox: List[float]) -> List[float]:
    """
    Converts COCO format bounding box to DOTA format coordinates.
    
    Args:
        coco_bbox: COCO format bounding box [x_min, y_min, width, height]
                  where all values are in absolute coordinates.
    
    Returns:
        List of 8 coordinates in DOTA format [x1, y1, x2, y2, x3, y3, x4, y4]
        representing the four corners in clockwise order.
    
    Examples:
        >>> coco_bbox = [100, 100, 50, 30]  # x_min, y_min, width, height
        >>> convert_coco_to_dota(coco_bbox)
        [100, 100, 150, 100, 150, 130, 100, 130]  # Four corners
    """
    x_min, y_min, width, height = coco_bbox
    x1, y1 = x_min, y_min
    x2, y2 = x_min + width, y_min
    x3, y3 = x_min + width, y_min + height
    x4, y4 = x_min, y_min + height
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def check_image_size(
    img_width: int, 
    img_height: int, 
    target_width: int, 
    target_height: int
) -> str:
    """
    Validates image dimensions against target size and determines if padding is needed.
    
    Args:
        img_width: Current image width in pixels.
        img_height: Current image height in pixels.
        target_width: Desired target width in pixels.
        target_height: Desired target height in pixels.
    
    Returns:
        'valid' if image matches target size, 'pad' if padding is needed.
    
    Raises:
        ValueError: If image size exceeds target size.
    
    Examples:
        >>> check_image_size(800, 600, 1024, 1024)
        'pad'
        >>> check_image_size(1024, 1024, 1024, 1024)
        'valid'
        >>> check_image_size(1200, 1024, 1024, 1024)
        ValueError: Image size (1200x1024) exceeds target size (1024x1024).
    """
    if img_width > target_width or img_height > target_height:
        raise ValueError(f"Image size ({img_width}x{img_height}) exceeds target size ({target_width}x{target_height}).")
    elif img_width < target_width or img_height < target_height:
        return 'pad'
    return 'valid'


def random_pad_image_and_labels(  # renamed to remove "rel" since it handles both
    image: Image.Image,
    labels: List[List[float]],
    target_width: int,
    target_height: int,
    random_seed: int = RANDOM_SEED
) -> Tuple[Image.Image, List[List[float]]]:
    """
    Randomly pads an image and adjusts label coordinates.
    Handles both relative and absolute coordinates.
    
    Args:
        image: PIL Image to be padded
        labels: List of labels. Each label is [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
               Coordinates can be either relative (0-1) or absolute
        target_width: Desired width after padding
        target_height: Desired height after padding
        random_seed: Random seed for reproducible padding
    
    Returns:
        Tuple of (padded_image, updated_labels) where labels are in relative coordinates
    """
    random.seed(random_seed)
    
    img_width, img_height = image.size

    # Calculate padding
    pad_width = target_width - img_width
    pad_height = target_height - img_height
    left = random.randint(0, pad_width)
    right = pad_width - left
    top = random.randint(0, pad_height)
    bottom = pad_height - top

    # Pad the image
    padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    # Update labels
    updated_labels = []
    for label in labels:
        class_id = label[0]
        coords = label[1:]
        
        # Detect coordinate format
        coord_format = detect_coordinate_format(coords, img_width, img_height)
        
        if coord_format == 'relative':
            # Convert to absolute first
            abs_coords = convert_relative_to_absolute(coords, img_width, img_height)
        else:
            abs_coords = coords

        # Adjust absolute coordinates for padding
        padded_coords = [
            abs_coords[i] + (left if i % 2 == 0 else top)
            for i in range(len(abs_coords))
        ]

        # Convert back to relative coordinates using new dimensions
        rel_coords = convert_absolute_to_relative(padded_coords, target_width, target_height)
        updated_labels.append([class_id] + rel_coords)

    return padded_image, updated_labels

def tile_image_and_labels(
    image: Image.Image,
    labels: List[List[float]],
    tile_size: Tuple[int, int],
    overlap: float = 0.0
) -> List[Tuple[Image.Image, List[List[float]], Tuple[int, int]]]:
    """
    Tiles large images and adjusts label coordinates.
    Handles both relative and absolute coordinates.
    
    Args:
        image: PIL Image to be tiled
        labels: List of labels. Each label is [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
               Coordinates can be either relative (0-1) or absolute
        tile_size: Desired (width, height) of tiles
        overlap: Fraction of overlap between tiles (0-1)
    
    Returns:
        List of (tiled_image, adjusted_labels, (tile_x, tile_y)) tuples
        where adjusted_labels are in relative coordinates
    """
    img_width, img_height = image.size
    tile_width, tile_height = tile_size
    
    # Calculate stride
    stride_x = int(tile_width * (1 - overlap))
    stride_y = int(tile_height * (1 - overlap))
    
    # Calculate number of tiles needed
    n_tiles_x = math.ceil((img_width - tile_width) / stride_x) + 1
    n_tiles_y = math.ceil((img_height - tile_height) / stride_y) + 1
    
    tiles = []
    
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # Calculate tile boundaries
            x1 = tx * stride_x
            y1 = ty * stride_y
            x2 = min(x1 + tile_width, img_width)
            y2 = min(y1 + tile_height, img_height)
            
            # Extract tile
            tile = image.crop((x1, y1, x2, y2))
            
            # If tile is smaller than target size, pad it
            if tile.size != tile_size:
                tile_temp = Image.new(image.mode, tile_size, 0)
                tile_temp.paste(tile, (0, 0))
                tile = tile_temp
            
            # Adjust labels for this tile
            tile_labels = []
            for label in labels:
                class_id = label[0]
                coords = label[1:]
                
                # Detect coordinate format
                coord_format = detect_coordinate_format(coords, img_width, img_height)
                
                if coord_format == 'relative':
                    # Convert to absolute for processing
                    abs_coords = convert_relative_to_absolute(coords, img_width, img_height)
                else:
                    abs_coords = coords
                
                # Create points list for intersection check
                box_points = [(abs_coords[i], abs_coords[i+1]) 
                            for i in range(0, len(abs_coords), 2)]
                tile_box = (x1, y1, x2, y2)
                
                # Check if box intersects with tile
                if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in box_points):
                    # Adjust coordinates relative to tile
                    adj_coords = []
                    for i in range(0, len(abs_coords), 2):
                        adj_x = (abs_coords[i] - x1) / tile_width
                        adj_y = (abs_coords[i+1] - y1) / tile_height
                        adj_coords.extend([adj_x, adj_y])
                    
                    # Clip coordinates to [0, 1]
                    adj_coords = [min(max(c, 0), 1) for c in adj_coords]
                    tile_labels.append([class_id] + adj_coords)
            
            if tile_labels:  # Only add tiles that contain objects
                tiles.append((tile, tile_labels, (tx, ty)))
    
    return tiles

def process_image_and_labels(
    image: Image.Image,
    labels: List[List[float]],
    target_width: int,
    target_height: int,
    tile_if_large: bool = True
) -> Union[Tuple[Image.Image, List[List[float]]], 
          List[Tuple[Image.Image, List[List[float]], Tuple[int, int]]]]:
    """
    Enhanced version that handles both padding and tiling.
    
    Args:
        image: PIL Image to process
        labels: List of labels with relative coordinates
        target_width: Desired width
        target_height: Desired height
        tile_if_large: Whether to tile images larger than target size
    
    Returns:
        Either (processed_image, processed_labels) if padding is used,
        or list of (tiled_image, tile_labels, (tile_x, tile_y)) if tiling is used
    """
    img_width, img_height = image.size
    
    if img_width > target_width or img_height > target_height:
        if tile_if_large:
            return tile_image_and_labels(
                image, 
                labels, 
                (target_width, target_height)
            )
        else:
            # Resize if tiling is not desired
            scale = min(target_width/img_width, target_height/img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Adjust label coordinates for resizing
            adjusted_labels = []
            for label in labels:
                class_id = label[0]
                coords = label[1:]
                adjusted_coords = coords  # No adjustment needed for relative coords
                adjusted_labels.append([class_id] + adjusted_coords)
            
            return random_pad_image_and_labels(image, adjusted_labels, 
                                                 target_width, target_height)
    else:
        return random_pad_image_and_labels(image, labels, 
                                             target_width, target_height)

def generate_distinct_colors(num_classes: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct colors for different classes.
    Uses HSV color space to generate well-distributed colors.
    
    Args:
        num_classes: Number of distinct colors needed
        seed: Random seed for reproducibility
    
    Returns:
        List of RGB color tuples
    """
    np.random.seed(seed)
    
    colors = []
    for i in range(num_classes):
        # Use golden ratio to get well-distributed hues
        hue = i * 0.618033988749895 % 1.0
        
        # Fixed saturation and value for good visibility
        saturation = 0.7 + np.random.random() * 0.3  # 0.7-1.0
        value = 0.8 + np.random.random() * 0.2  # 0.8-1.0
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
            
        # Convert to 0-255 range
        rgb = tuple(int((color + m) * 255) for color in rgb)
        colors.append(rgb)
    
    return colors

def visualize_labels(
    image: Union[str, Image.Image],
    labels: List[List[Union[str, int, float]]],  # Modified to accept string class names
    output_path: Optional[str] = None,
    thickness: int = 2,
    num_samples: int = None,
    class_colors: Optional[List[Tuple[int, int, int]]] = None,
    num_classes: int = 60,  # Default for xView
    random_seed: int = 42
) -> Union[Image.Image, List[Image.Image]]:
    """
    Visualizes bounding box labels on images with distinct colors for different classes.
    Now supports both integer class indices and string class names.
    """
    # Generate or validate colors
    if class_colors is None:
        class_colors = generate_distinct_colors(num_classes, random_seed)
    else:
        if not all(validate_rgb_color(color) for color in class_colors):
            raise ValueError("Invalid RGB colors in class_colors")

    # Create color hash for string class names
    def get_color_for_class(class_name):
        if isinstance(class_name, int):
            return class_colors[class_name % len(class_colors)]
        else:
            # Hash the string to get a consistent color
            hash_val = hash(class_name)
            return class_colors[abs(hash_val) % len(class_colors)]
    
    # Handle single image case
    if isinstance(image, (str, Image.Image)):
        return _visualize_single_image(image, labels, output_path, get_color_for_class, thickness)
    
    # Rest of the function remains the same...

def _visualize_single_image(
    image: Union[str, Image.Image],
    labels: List[List[Union[str, int, float]]],
    output_path: Optional[str],
    color_func: Callable,
    thickness: int
) -> Image.Image:
    """Helper function to visualize a single image with class-specific colors"""
    try:
        if isinstance(image, str):
            image = Image.open(image)
    except Exception as e:
        logging.error(f"Failed to open image: {str(e)}")
        raise IOError(f"Could not open image: {str(e)}")
    
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB')
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    img_width, img_height = image.size
    
    for label in labels:
        try:
            class_id = label[0]  # Can be string or int now
            coords = label[1:]
            
            if not validate_coordinates(coords):
                logging.warning(f"Skipping label with invalid coordinates: {coords}")
                continue
            
            # Get color for this class using the color function
            color = color_func(class_id)
            
            # Convert relative coordinates to absolute
            abs_coords = convert_relative_to_absolute(coords, img_width, img_height)
            
            # Draw the bounding box
            points = [
                (abs_coords[0], abs_coords[1]),
                (abs_coords[2], abs_coords[3]),
                (abs_coords[4], abs_coords[5]),
                (abs_coords[6], abs_coords[7]),
                (abs_coords[0], abs_coords[1])
            ]
            
            # Draw lines with class-specific color
            for i in range(4):
                draw.line([points[i], points[i+1]], fill=color, width=thickness)
            
            # Add class label with background
            label_pos = (abs_coords[0], abs_coords[1] - 20)
            label_text = str(class_id)  # Convert class ID to string
            
            # Draw text background with class color
            text_bbox = draw.textbbox(label_pos, label_text)
            draw.rectangle(
                [text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                fill=color
            )
            
            # Use black or white text depending on background color brightness
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            draw.text(label_pos, label_text, fill=text_color)
            
        except Exception as e:
            logging.error(f"Error processing label: {str(e)}")
            continue
    
    if output_path:
        try:
            img_draw.save(output_path)
        except Exception as e:
            logging.error(f"Failed to save image: {str(e)}")
            raise IOError(f"Could not save image: {str(e)}")
    
    return img_draw

def parse_shiprsimagenet_class(class_name: str) -> str:
    """
    Converts abbreviated ship class names to their full names with underscores.
    
    Args:
        class_name: Abbreviated or short class name from ShipRSImageNet_V1
        
    Returns:
        Full class name with abbreviations expanded and spaces replaced with underscores
    """
    # Define mapping of abbreviations to full names
    abbreviations = {
        "DD": "Destroyer",
        "FF": "Frigate",
        "LL": "Landing",
        "AS": "Auxiliary_Ship",
        "LSD": "Landing_Ship_Dock",
        "LHA": "Landing_Helicopter_Assault_Ship",
        "AOE": "Fast_Combat_Support_Ship",
        "EPF": "Expeditionary_Fast_Transport_Ship",
        "RoRo": "Roll_on_Roll_off_Ship"
    }
    
    # Replace abbreviations with full names
    full_name = class_name
    for abbr, full in abbreviations.items():
        full_name = full_name.replace(abbr, full)
    
    # Replace spaces with underscores
    return full_name.replace(" ", "_")

def read_imageset_file(file_path: str) -> List[str]:
    """
    Reads image set file (train.txt or val.txt) and returns list of basenames.
    
    Args:
        file_path: Path to the image set file
        
    Returns:
        List of basenames for the specified split (without file extensions)
    """
    with open(file_path, 'r') as f:
        # Remove any file extensions and whitespace
        return [os.path.splitext(line.strip())[0] for line in f.readlines()]
