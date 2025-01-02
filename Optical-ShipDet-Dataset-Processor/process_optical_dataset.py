from UnifiedDatasetProcessor import *

# Setup logging
setup_logging("kari_dataset_processing.log")

# Create configuration
config = ProcessingConfig(
    input_dir="/path/to/FAIR1M",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=True
)

# Create and use processor
try:
    processor = ProcessorFactory.create_processor("fair1m", config)
    processor.process()
except Exception as e:
    logging.error(f"Processing failed: {str(e)}")