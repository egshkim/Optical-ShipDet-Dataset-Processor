from UnifiedDatasetProcessor import *

config = ProcessingConfig(
    input_dir="/path/to/xview/dataset",
    output_dir="/path/to/output",
    target_size=(1024, 1024),
    coord_format=CoordinateFormat.RELATIVE,
    visualize=False
)
processor = DatasetProcessor(config)
processor.process_xview()