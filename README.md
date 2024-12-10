# Optical-ShipDet-Dataset-Processor

A unified tool for processing various optical satellite imagery ship detection datasets into a standardized format. This tool currently supports xView dataset processing, with robust error handling and comprehensive logging, focusing specifically on ship detection tasks.

## Features

- Unified processing pipeline for optical ship detection datasets
- Ship-specific object filtering and processing
- Support for relative coordinate formats
- Automatic image padding to target dimensions (default 1024x1024)
- Optional visualization of bounding box annotations with customizable colors and thickness
- Comprehensive error handling and detailed logging
- Reproducible random padding with configurable seed
- Configurable output formats and directories
- Robust coordinate validation and boundary checking
- Automatic image tiling for large satellite images
- Support for DOTA format annotations

## Supported Datasets

1. **xView**
   - Paper: [xView: Objects in Context in Overhead Imagery](https://arxiv.org/abs/1802.07856)
   - Dataset: [http://xviewdataset.org/](http://xviewdataset.org/)
   - Ship-related classes:
     ```python
     ship_classes = {
         23: "Maritime Vessel",
         24: "Motorboat",
         25: "Sailboat",
         26: "Tugboat",
         27: "Barge",
         28: "Fishing Vessel",
         29: "Ferry",
         30: "Yacht",
         31: "Container Ship",
         32: "Oil Tanker"
     }
     ```
   - GeoJSON format annotations
   - High-resolution satellite imagery
   - Conversion: GeoJSON format → DOTA format → Relative coordinates

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Project Structure

```
├── utils.py                    # Core utility functions for coordinate conversion and image processing
├── UnifiedDatasetProcessor.py  # Main dataset processing class
├── process_optical_dataset.py  # Example processing script
├── requirements.txt
└── outputs/
    ├── train/ 
        ├── images/            # Processed images
        ├── labels/            # Processed labels
        └── visualizations/    # Optional visualization outputs
```

## Usage

### Basic Usage with Processing Config

```python
from UnifiedDatasetProcessor import DatasetProcessor, ProcessingConfig, CoordinateFormat

# Configure the processor
config = ProcessingConfig(
    input_dir="/path/to/xview/dataset",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True  # Enable visualization
)

# Initialize and run processor
processor = DatasetProcessor(config)
processor.process_xview()  # Process xView dataset
```

### Advanced Configuration Options

```python
@dataclass
class ProcessingConfig:
    input_dir: str                    # Input dataset directory
    output_dir: str                   # Output directory for processed files
    target_size: Tuple[int, int]      # Target dimensions (width, height)
    coord_format: CoordinateFormat    # RELATIVE or ABSOLUTE
    visualize: bool = False           # Enable visualization
    visualization_dir: Optional[str] = None  # Custom visualization directory
```

## Input Dataset Requirements

### xView Dataset Structure
```
xview_dataset/
├── train_images/
├── train_labels/
│   └── xView_train.geojson
└── val_images/
```

## Key Features Implementation

### Ship-Specific Processing
- Filters and processes only ship-related objects from the dataset
- Skips images and tiles without ship objects
- Maintains original ship class information

### Error Handling and Logging
- Comprehensive error catching and reporting
- Detailed logging with timestamps
- Graceful handling of invalid coordinates
- Warnings for potential issues

## Output Format
Processed dataset follows a standardized format:
```
output_dir/
    ├── train/
        ├── images/
        │   └── xView_{original_filename}.png
        ├── labels/
        │   └── xView_{original_filename}.txt
        └── visualizations/
            └── xView_{original_filename}_vis.png
```

### Label Format
The processed labels are stored in DOTA format with relative coordinates:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
where:
- `class_id`: The mapped ship class index (23-32)
- `x1,y1,x2,y2,x3,y3,x4,y4`: Normalized coordinates (0-1) of the bounding box corners in clockwise order

## Citation

If you use this processor in your research, please cite:

```bibtex
@software{Optical-ShipDet-Dataset-Processor,
  title={Optical-ShipDet-Dataset-Processor},
  author={SeonHoon Kim},
  year={2024},
  description={A unified tool for processing optical satellite imagery ship detection datasets}
}
```

## Future Work
- Add support for additional optical satellite imagery datasets
- Implement data augmentation techniques specific to ship detection
- Add support for instance segmentation annotations
- Extend visualization capabilities for rotated bounding boxes
