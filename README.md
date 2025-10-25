## About
BOB (Bounding-box Oracle for Biomedicine) is a prompt generator designed to be used with the MedSAM2 model.
It is trained on medical images, videos and 3D data to generate prompts for segmentation tasks.
This class provides methods to generate different types of prompts and assign classes to them.
For 3D images, it generates bounding boxes for every slice in the volume which then
get de-duplicated using non-maximum suppression of the confidence scores for each class.

## Installation and Downloads

You can install `BOB` via pip:

```bash
pip install -e src/segFM/BOB
```

Dependencies:
- napari
- numpy
- torch
- MedVol
- ultralytics

For ease of use, you can also use the Makefile provided in the segFM root directory.
Read the [main README](../../../README.md) for more information.

Some example images from multiple datasets can be [downloaded from Google Drive here](https://drive.google.com/drive/folders/14b9zoYAizITFiuQ92DeSbv4t5dGpsVol?usp=drive_link).

### Optional Dependencies

For video file support (MP4, AVI, MOV, MKV), install with the video extra:

```bash
pip install -e "src/segFM/BOB[video]"
```

This installs imageio with ffmpeg support for reading video files. Videos are automatically converted to grayscale for processing.


## Usage

1. Open napari in your Python environment `python -m napari`
2. Load your medical image data
3. Go to `Plugins > BOB (Bounding-box Oracle for Biomedicine)` to open the BOB widget
4. Configure parameters:
    - `Preset`: Choose a preset configuration for easy prompt generation
    - `Prompt Generation Model`: Select the model to use for prompt generation, e.g. YOLOv12n or D-FINE-N
    - `Confidence Threshold`: Minimum confidence score for detected objects (0.0-1.0)
    - `IoU Threshold`: Intersection over Union threshold for non-maximum suppression of YOLO models (0.0-1.0)
    - `Max Detections`: Maximum number of objects to detect per image (1-100)
    - `Prompts per Object`: Number of prompts to generate per object (=same object_id) in 3D
    - `Multi-prompt Z spacing`: Minimum distance between prompts for the same object in z-direction (after 3D clustering)
    - `Max Z Distance`: Maximum distance prompts can be apart in z-direction to be considered the same object (during 3D clustering)
    - `Allow Multiobject 3D`: Whether to allow multiple objects of the same type in 3D (otherwise all are assigned the same object_id)
    - `Predict all classes`: Whether the model should predict all supported classes or only the selected ones

5. Select your preferred Prompt Generation Model or preset and click "Generate Prompts" to create prompts for the active image layer
6. Click "Clear Prompts" to remove all generated prompt visualizations
7. Add manual box prompts by clicking "Add Prompt manually" and drawing boxes on the image
8. Select your preferred segmentation model in the dropdown menu and click on "Segment Prompts" to segment the currently active image layer using all available prompts

## Features

- Fast and lightweight prompt generation + segmentation, suitable for real-time applications
- Supports CPU, GPU and MPS (Apple Silicon) and automatically selects the best available device
- Interactive widget for prompt generation parameter configuration
- Support for both 2D and 3D medical images
- Visual prompt representation using napari Shapes layers
- Color-coded prompts by object ID
- Easy clearing of generated prompts