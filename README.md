# Optical-ShipDet-Dataset-Processor

A unified tool for processing various optical satellite imagery ship detection datasets into a standardized format. This tool supports processing of multiple satellite imagery datasets, with robust error handling and comprehensive logging, focusing specifically on ship detection tasks.

## Features
- Process multiple satellite imagery ship detection datasets into standardized format
- Support for DOTA, xView, FAIR1M, ShipRSImageNet_V1, KARI-Ship-Det-Beta, and Airbus Ship Detection Challenge datasets
- Convert various annotation formats to DOTA format with relative coordinates
- Automatic image resizing and tiling with configurable settings
- Optional visualization of processed results
- Extensible architecture for adding new datasets

## Supported Datasets

### 1. DOTA (Dataset for Object deTection in Aerial images)
- Paper: [DOTA: A Large-scale Dataset for Object Detection in Aerial Images](https://arxiv.org/abs/1711.10398)
- Dataset: [https://captain-whu.github.io/DOTA/](https://captain-whu.github.io/DOTA/)
- Features:
  - 15 common object categories
  - Over 188k instances
  - Oriented bounding box annotations
- Usage Example:
```python
config = ProcessingConfig(
    input_dir="/path/to/DOTA",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

processor = ProcessorFactory.create_processor("dotav15", config)
processor.process()
```

### 2. xView
- Paper: [xView: Objects in Context in Overhead Imagery](https://arxiv.org/abs/1802.07856)
- Dataset: [http://xviewdataset.org/](http://xviewdataset.org/)
- Features:
  - 60 object classes including maritime vessels
  - Over 1 million instances
  - High-resolution satellite imagery
- Usage Example:
```python
config = ProcessingConfig(
    input_dir="/path/to/xView",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

processor = ProcessorFactory.create_processor("xview", config)
processor.process()
```

### 3. FAIR1M
- Paper: [FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery](https://arxiv.org/abs/2103.05569)
- Dataset: [http://gaofen-challenge.com/](http://gaofen-challenge.com/)
- Features:
  - 37 fine-grained categories
  - Over 1 million instances
  - Rich geographic information
- Usage Example:
```python
config = ProcessingConfig(
    input_dir="/path/to/FAIR1M",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

processor = ProcessorFactory.create_processor("fair1m", config)
processor.process()
```

### 4. ShipRSImageNet_V1
- Paper: [https://ieeexplore.ieee.org/document/9512396/](https://ieeexplore.ieee.org/document/9512396/)
- Dataset: [https://github.com/zzndream/ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet)
- Features:
  - Detailed ship categories
  - VOC format annotations
  - High-resolution ship imagery
- Usage Example:
```python
config = ProcessingConfig(
    input_dir="/path/to/ShipRSImageNet_V1",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

processor = ProcessorFactory.create_processor("shiprsimagenet", config)
processor.process()
```

### 5. KARI Ship Detection Beta

- Paper: [(https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11813752)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11813752)
- Dataset: [https://doi.org/10.22711/idr/1021](https://doi.org/10.22711/idr/1021)

- Features:
  - KOMPSAT-3/3A satellite imagery
  - Ship instance annotations
  - Multiple resolutions
- Usage Example:
```python
config = ProcessingConfig(
    input_dir="/path/to/kari-ship-det-beta",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

processor = ProcessorFactory.create_processor("kari", config)
processor.process()
```

### 6. Airbus Ship Detection Challenge
- Competition: [https://www.kaggle.com/c/airbus-ship-detection](https://www.kaggle.com/c/airbus-ship-detection)
- Features:
  - 192,556 images with 81,723 ships
  - Run-length encoded masks
  - Binary ship/no-ship classification
- Usage Example:
```python
config = ProcessingConfig(
    input_dir="/path/to/airbus-ship-detection",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

processor = ProcessorFactory.create_processor("airbus", config)
processor.process()
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Project Structure

```
├── utils.py                    # Core utility functions
├── UnifiedDatasetProcessor.py  # Main processor implementation
├── requirements.txt
└── outputs/
    ├── train/ 
    │   ├── images/            # Processed images
    │   ├── labels/            # Processed labels
    │   └── visualizations/    # Optional visualization outputs
    └── val/                   # Same structure as train/
```

## Output Format
Processed dataset follows a standardized format:
```
output_dir/
├── train/
│   ├── images/
│   │   └── {dataset}_{original_filename}.png
│   ├── labels/
│   │   └── {dataset}_{original_filename}.txt
│   └── visualizations/
│       └── {dataset}_{original_filename}_vis.png
└── val/
    └── ... (same structure as train/)
```

### Label Format
The processed labels are stored in DOTA format with relative coordinates:
```
class_name x1 y1 x2 y2 x3 y3 x4 y4
```
where:
- `class_name`: String identifier for the ship class
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
- Add support for instance segmentation annotations
