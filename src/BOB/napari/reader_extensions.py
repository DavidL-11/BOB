import numpy as np
from medvol import MedVol

# Optional video support - imageio with ffmpeg
try:
    import imageio.v3 as iio
    HAS_VIDEO_SUPPORT = True
except ImportError:
    HAS_VIDEO_SUPPORT = False
    iio = None

def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.
    
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
        
    Returns
    -------
    function or None
        If the path is a supported format, return a reader function.
        Otherwise, return None.
    """
    if isinstance(path, list):
        path = path[0]
    
    # Check if it's a supported video format
    if isinstance(path, str) and path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Check if video support is available
        if not HAS_VIDEO_SUPPORT:
            return None
        return video_reader_function
    elif (path.endswith(".nii") or path.endswith(".nii.gz") or path.endswith(".nrrd")):
        return medical_image_reader_function
    return None

# Taken from napari-nifti / https://github.com/MIC-DKFZ/napari-nifti/blob/main/src/napari_nifti/_reader.py
def medical_image_reader_function(path):
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files
    image_data_list = [MedVol(_path) for _path in paths]
    # Convert to LayerData tuples
    layer_data = [(image_data.array, {"affine": image_data.affine, 
                                      "metadata": {"spacing": image_data.spacing, "origin": image_data.origin, "direction": image_data.direction, "header": image_data.header}}, "image")
                  for image_data in image_data_list]
    return layer_data

def video_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.
    
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
        
    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, [metadata], layer_type), where data is a numpy array,
        metadata is a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a string.
    """
    if not HAS_VIDEO_SUPPORT:
        raise ImportError(
            "Video support not available. Install with video support using: "
            "pip install 'BOB[video]'"
        )
    
    try:
        # Read video into array: (frames, height, width, channels)
        frames = list(iio.imiter(path))
        data = np.stack(frames, axis=0)
        
        # Ensure data is in the right format
        if data.ndim == 4 and data.shape[-1] in [3, 4]:  # RGB or RGBA
            # For RGB/RGBA videos, convert to grayscale
            data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
        elif data.ndim == 4 and data.shape[-1] == 1:  # Grayscale with channel
            # Remove the single channel dimension
            data = data.squeeze(-1)
        elif data.ndim == 3:  # Already grayscale (T, Y, X)
            pass
        else:
            raise ValueError(f"Unexpected video shape: {data.shape}")
        
        # Return in napari's expected format: list of layer data tuples
        # The metadata dict can contain additional parameters for the layer
        metadata = {
            "name": f"video_{path.split('/')[-1]}",
            "metadata": {"video_path": path}
        }
        
        return [(data, metadata, "image")]
        
    except Exception as e:
        print(f"Error reading video file {path}: {e}")
        return None