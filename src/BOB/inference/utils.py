import numpy as np
import torch
from PIL import Image

from BOB.logger import logger

class Prompt(object):
    """
    A class to represent a prompt for a segmentation model.
    The prompt can be a bounding box, a point, or both.
    All points and boxes contained in this prompt represent the same object.
    Every prompt can have max. one bounding box -> 1D-Array (Example: p.box = [x1, y1, x2, y2]).
    Every prompt can have multiple point prompts -> 2D-Array (Example: p.point = [[x1, y1], [x2, y2], ...]).
    Every point prompt has a label (1 for foreground, 0 for background) -> 1D-Array (Example: p.label = [1, 0, 1, ...]).
    """

    def __init__(
        self,
        box=None,
        point=None,
        label=None,
        z=0,
        confidence=1.0,
        quality_score=1.0,
        color=255,
        channel=None,
        class_label="",
        class_id=None,
        generated_by="",
        obj_id=-1,
    ):
        """
        Initializes the Prompt object with a bounding box, point, label, object ID, z-coordinate, quality score, class label,
          and generation method.

        Parameters:
            box (numpy.ndarray): The bounding box coordinates in the format [x1, y1, x2, y2].
            point (numpy.ndarray): The point coordinates in the format [x, y].
            label (numpy.ndarray): The label for the point (1 for foreground, 0 for background).
            z (int): The z-coordinate for 3D prompts.
            confidence (float): A score representing the confidence of the prompt (For yolo prompts)
            quality_score (float): A score representing the quality of the prompt (For yolo prompts)
            color (int): The color the object would have in the ground truth mask (to calculate metrics).
            channel (int): The channel of the labelmap this prompt was generated from (if applicable).
            class_id (int): The class ID of the object (for BOB training).
            class_label (str): The class label of the object (for human readability).
            generated_by (str): The method used to generate the prompt.
            obj_id (int): A unique ID for the object this prompt belongs to. Needed for multi-prompt scenarios.
        """
        self.box = box
        self.point = point
        self.label = label
        self.z = z
        self.confidence = confidence # Between 0 and 1, used for yolo prompts
        self.quality_score = quality_score # Calculated from confidence and area of the bounding box
        self.color = color
        self.channel = channel  # The channel is for prompts generated from a labelmap - in this case color=None, otherwise channel=None
        self.class_label = class_label
        self.generated_by = generated_by
        self.class_id = class_id
        self.obj_id = obj_id  # Unique ID for the object this prompt belongs to

    def is_empty(self):
        """
        Checks if the prompt is empty, meaning it has no bounding box or point.

        Returns:
            bool: True if both box and point are None, False otherwise.
        """
        return self.box is None and self.point is None

    def get_best(self):
        """
        Returns the best prompt based on the presence of a bounding box or point.
        If both are present, the bounding box is preferred.
        """
        if self.box is not None:
            return self.box
        elif self.point is not None:
            return self.point
        else:
            raise ValueError("No prompt available. Both box and point are None.")

    def get_box(self):
        """
        Returns the bounding box coordinates.
        """
        return self.box

    def get_point(self):
        """
        Returns the point coordinates.
        """
        return self.point
    
def normalize_minmax(img):
    """
    Normalize a NumPy image array to the range [0, 255],
    similar to cv2.normalize with NORM_MINMAX.
    """
    img = img.astype(np.float32)  # ensure float for division
    min_val = np.min(img)
    max_val = np.max(img)

    # avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    norm_img = (img - min_val) * 255.0 / (max_val - min_val)
    return norm_img.astype(np.uint8)

def nifti_to_numpy(filepath):
    """
    Loads a NIfTI file and converts it to a NumPy array in SAR orientation (Slice, Anterior, Right).
    If the data has multiple channels, only the first channel is retained.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to load NIfTI files. Please install it via 'pip install nibabel'.")
    
    nii = nib.load(filepath)
    nii_ras = nib.as_closest_canonical(nii)
    data_ras = nii_ras.get_fdata()

    # If the data is in the shape (x, y, z, channels), like MedicalDecathlon, remove the last dimension
    if data_ras.ndim == 4 and data_ras.shape[-1] >= 1:
        data_ras = data_ras[..., 0]

    # If the data is not in the range of 0-255, normalize it
    if min(np.unique(data_ras)) < 0 or max(np.unique(data_ras)) > 255:
        data_ras = normalize_minmax(data_ras)

    # Convert to SAR by swapping axes
    data_ras = np.transpose(data_ras, (2, 1, 0))  # Change from (x, y, z) to (z, y, x)
    data_ras = np.ascontiguousarray(data_ras, dtype=np.uint8)

    return data_ras


def grayscale_to_rgb(image):
    """
    Converts a grayscale image to RGB format by stacking the grayscale image across three channels.
    
    Parameters:
        image (numpy.ndarray): The input grayscale image.
        
    Returns:
        numpy.ndarray: The RGB image.
    """
    if image.shape[-1] != 1:
        # If the image is of shape (H, W) or (D, H, W) for 3D images, stack it at the last dimension
        return np.stack((image,)*3, axis=-1)
    else: 
        # If grayscale image is in shape (H, W, 1), convert it to (H, W, 3)
        return np.concatenate([image]*3, axis=-1)


def setup_device() -> torch.device:
    """
    Sets up the GPU device for computation.
    It checks if CUDA or MPS is available and sets the device accordingly.
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda").__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        logger.warning(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    return device

def create_binary_segmentation(mask):
    """
    Creates a binary image from the predicted mask.

    Parameters:
        mask (numpy.ndarray): The predicted mask to be converted to binary.
    Returns:
        numpy.ndarray: The binary image.
    """

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w)

    # Ensure the mask is binary
    mask_image = (mask_image > 0).astype(np.uint8) * 255

    return mask_image


def resize_and_normalize(image):
    """
    This function resizes a 3D grayscale numpy array to a 4D numpy array of shape (d, 3, image_size, image_size)
    and normalizes it to the range [0, 1] with ImageNet mean and std.
    The input image is expected to be in the shape (d, h, w).

    Args:
        image (numpy.ndarray): Input 3D grayscale image array of shape (d, h, w).
    Returns:
        torch.Tensor: A 4D tensor of shape (d, 3, image_size, image_size) with normalized pixel values.
    """
    img_resized = transpose_and_resize(image, 512)

    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean = (0.485, 0.456, 0.406)  # ImageNet mean
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    return img_resized

def transpose_and_resize(array, image_size):
    """
    Resize a 3D grayscale numpy array to a 4D numpy array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    array = grayscale_to_rgb(array)  # Ensure the array is in RGB format
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(
            2, 0, 1
        )  # (3, image_size, image_size)
        resized_array[i] = img_array

    return resized_array