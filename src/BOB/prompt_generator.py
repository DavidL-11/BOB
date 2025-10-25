import numpy as np
import torch
from PIL import Image
from collections import defaultdict, deque
from typing import List
import os

from BOB.inference import utils

def bounding_box_iou(boxA, boxB):
    """
    Computes IntersectionOverUnion between two boxes in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    union_area = boxA_area + boxB_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou

class BOB:
    """
    BOB (Bounding-box Oracle for Biomedicine) is a prompt generator designed to be used with the MedSAM2 model
    or other prompt-based segmentation models designed for medical imaging.
    It is trained on medical images, videos and 3D data to generate prompts for segmentation tasks.
    This class provides methods to generate different types of prompts and assign classes to them.
    For 3D images, it generates bounding boxes for every slice in the volume which then
    get de-duplicated using non-maximum suppression of the confidence scores.

    It is possible to switch between different models, such as YOLOv12n or D-FINE-N.
    Note that the model has to follow the YOLO-style interface, i.e. it should have a
    - `predict` method that returns an list of objects that have a 'boxes' attribute
    - `names` attribute that contains the class names by object ID
    """

    def __init__(
        self,
        model="D-FINE-N", # Options: "YOLOv12n", "YOLOv12s", "D-FINE-N"
    ):
        """
        Initialize the Bounding-box Oracle for Biomedicine (BOB) with the specified mode.

        Args:
            mode (str): The mode of prompt generation. Options are "box", "point", or "yolo".
        """
        self.switch_model(model)

    def get_supported_classes(self):
        """
        Returns the list of classes supported by the YOLO model.
        This is useful for understanding what objects can be detected in the images.
        """
        return self.model.names
    
    def switch_model(self, model):
        """
        Switch the model used for prompt generation.
        Args:
            model (str): The name of the model to switch to. Options are "YOLOv12n", "YOLOv12s", or "D-FINE-N".
        """
        file_location = os.path.dirname(os.path.abspath(__file__))

        if model == "YOLOv12n" or model == "YOLOv12-Nano":
            from ultralytics import YOLO
            self.model = YOLO(os.path.join(file_location, "checkpoints/YOLOv12n/YOLOv12n.pt"))
        elif model == "YOLOv12s" or model == "YOLOv12-Small":
            from ultralytics import YOLO
            self.model = YOLO(os.path.join(file_location, "checkpoints/YOLOv12s/YOLOv12s.pt"))
        elif model == "D-FINE-N" or model == "D-FINE-Nano":
            # from src.BOB.inference.dfine_predictor import DFinePredictor # For inference using original torch model
            from BOB.inference.dfine_onnx import DFinePredictor
            self.model = DFinePredictor(os.path.join(file_location, "checkpoints/D-FINE-N/D-FINE-N.onnx"))
        else:
            raise ValueError("Invalid model specified. Choose 'YOLOv12n', 'YOLOv12s', or 'D-FINE-N'.")

    def generate_prompt(
        self, 
        img, 
        dataset,
        confidence=0.5,
        iou=0.7,
        max_det=20,
        n_prompts_per_obj=5, 
        multiprompt_z_spacing=10,
        max_z_distance=10,
        plot_prompts=False, 
        allow_multiobject_3d=True,
        allowed_classes=None
    ):
        """
        This method generates a prompt for the given image based on the mode.
        The image can be a either 2D or 3D. Accepted are numpy arrays, PIL images, file paths (jpg, jpeg, png, nii.gz)
        or tensor images.
        Args:
            img (str, numpy.ndarray, PIL.Image, torch.Tensor): The input image for which to generate a prompt.
            dataset (BaseImageDataset): The dataset to which the image belongs. Used for color mapping (id_to_color).
            n_prompts_per_obj (int): Number of prompts to generate per object.
            min_z_distance (int): Minimum distance between the n prompts generated for each object.
            plot_prompts (bool): Whether to plot the generated prompts on the image.
        Returns:
            list: A list of prompts generated for the image.
        """

        # Convert img/path to numpy array
        img_processed = self.__load_image__(img)

        # Convert grayscale to RGB if necessary
        if img_processed.ndim == 2 or (img_processed.ndim == 3 and img_processed.shape[-1] != 3):
            img_processed = utils.grayscale_to_rgb(img_processed)

        if img_processed.ndim == 3 and img_processed.shape[-1] == 3:
            return self.__box_prompt_2d__(
                img_processed,
                plot_prompts=plot_prompts,
                conf=confidence,
                iou=iou,
                max_det=max_det,
                dataset=dataset,
                allowed_classes=allowed_classes,
            )
        elif img_processed.ndim == 4 and img_processed.shape[-1] == 3:
            return self.__box_prompt_3d__(
                img_processed,
                plot_prompts=plot_prompts,
                conf=confidence,
                iou_thresh=iou,
                dataset=dataset,
                n_prompts_per_obj=n_prompts_per_obj,
                multiprompt_z_spacing=multiprompt_z_spacing,
                z_dist_max=max_z_distance,
                max_det=max_det,
                allowed_classes=allowed_classes,
                allow_multiobject_3d=allow_multiobject_3d,
            )
        else:
            raise ValueError(
                f"Unsupported image dimensions for image of shape {img_processed.shape}. "
                "Image must be 2D grayscale or RGB, or 3D grayscale"
            )

    def __load_image__(self, img):
        """
        Load the image from various formats (file path, numpy array, PIL image, or tensor).

        Args:
            img (str, numpy.ndarray, PIL.Image, torch.Tensor): The input image.

        Returns:
            numpy.ndarray: The loaded image as a numpy array.
        """
        if isinstance(img, str):
            if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                # Load from file path
                img = Image.open(img).convert("RGB")
                img = np.array(img)
            elif img.endswith(".nii.gz"):
                img = utils.nifti_to_numpy(img)
                print("Loaded NIfTI image with shape:", img.shape)
                print("Converted NIfTI image to RGB with shape:", img.shape)
        elif isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
            # If the data is not in the range of 0-255, normalize it
            if min(np.unique(img)) < 0 or max(np.unique(img)) > 255:
                img = utils.normalize_minmax(img)
        elif isinstance(img, Image.Image):
            # Convert PIL image to numpy array
            img = np.array(img.convert("RGB"))
        else:
            raise TypeError(
                "Unsupported image format. Provide a file path, numpy array, PIL image, or tensor."
            )
        return img

    def __convert_dict_to_array__(self, data):
        return np.array(
            [
                utils.Prompt(
                    box=np.array(bbox),
                    z=z,
                    class_label=cls,
                )
                for cls, arr in data.items()
                for bbox, _, z in arr
            ],
            dtype=object,
        )

    def __box_prompt_2d__(self, img, conf, iou, max_det, dataset, plot_prompts=False, allowed_classes=None):
        """
        Generate bounding box prompts for a 2D image using YOLO.
        Args:
            img (numpy.ndarray): The input 2D image.
            conf (float): Confidence threshold for YOLO detection. Higher values mean fewer but higher quality detections.
            iou (float): IoU threshold for non-max suppression. Lower values mean less overlap allowed between boxes -> less boxes.
            max_det (int): Maximum number of detections per image.
        Returns:
            list: A list of tuples containing bounding boxes, quality scores, and class labels.
            Each tuple is in the format (bbox, quality_score, class_label).
        """
        results = self.model.predict(
            source=img,  # Use the image from the dataset
            imgsz=512,  # Resize image to 512x512 for YOLO
            conf=conf,  # Confidence threshold
            iou=iou,  # IoU threshold for non-max suppression
            max_det=max_det,  # Maximum number of detections per image
            show=plot_prompts,  # Show the prompts on the image
            classes=allowed_classes,  # Filter by allowed classes if provided
            device=(
                utils.setup_device()
            ),  # Use GPU if available
            verbose=False,  # Disable verbose output
        )

        # Get the bounding boxes from the results
        bboxes = results[0].boxes.xyxy.cpu().numpy()

        # Convert the boxes to integer format
        bboxes = bboxes.astype(int)

        # Get the confidence scores
        scores = results[0].boxes.conf.cpu().numpy()

        # Convert the confidence scores to quality scores (depending on score + area)
        quality_scores = self.__box_quality__(bboxes, scores)

        # Get the class labels as a string for better readability
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        combined = zip(bboxes, quality_scores, scores, class_ids)

        # Combine the bounding boxes, scores, and class labels into a list of tuples
        res = [
            utils.Prompt(
                box=np.array(bbox),
                quality_score=quality_score,
                confidence=confidence,
                class_label=self.model.names[class_id],
                class_id=class_id,
                obj_id=i+1,  # Assign a unique object ID starting from 1
                color=dataset.id_to_color.get(class_id, 255) if dataset else None,
                generated_by="MedPAM",
            )
            for i, (bbox, quality_score, confidence, class_id) in enumerate(combined)
        ]

        return res

    def __box_prompt_3d__(
        self,
        img: np.ndarray,
        conf: float,
        iou_thresh: float,
        max_det: int,
        dataset: str,
        n_prompts_per_obj: int = 1,
        multiprompt_z_spacing: int = 10,
        z_dist_max: int = 10,
        plot_prompts: bool = False,
        allowed_classes: List[int] = None,
        allow_multiobject_3d: bool = True
    ) -> List[utils.Prompt]:
        """
        Generate and cluster 2D prompts into 3D objects by class, using IoU and z-spacing constraints.
        """

        D = img.shape[0]
        prompts_by_class = defaultdict(list)

        # --- Step 0: Preprocess image ---
        # uint8 conversion
        img = img.astype(np.uint8)

        # --- Step 1: Gather 2D prompts per slice and group by class ---
        for z in range(D):
            slice_img = img[z, :, :]
            prompts2d = self.__box_prompt_2d__(
                slice_img, conf=conf, iou=iou_thresh,
                max_det=max_det, dataset=dataset,
                plot_prompts=False,
                allowed_classes=allowed_classes
            )
            for p in prompts2d:
                p.z = z
                prompts_by_class[p.class_label].append(p)

        # --- Step 2: Assign obj_ids based on IoU + z-distance ---
        current_obj_id = 1
        for prompts_for_single_class in prompts_by_class.values():
            remaining_prompts = set(prompts_for_single_class)
            
            # Simple case: all prompts of same class get same obj_id
            if not allow_multiobject_3d:
                for p in remaining_prompts:
                    p.obj_id = current_obj_id
                current_obj_id += 1
                continue
            
            # Complex case: cluster by IoU + z-distance using BFS-style clustering
            while remaining_prompts:
                start_prompt = remaining_prompts.pop()
                start_prompt.obj_id = current_obj_id

                # The queue will hold prompts that are neighbors (= same object)
                queue = deque([start_prompt])
                while queue:
                    p = queue.popleft()
                    # Convert to list once to avoid repeated conversion
                    remaining_list = list(remaining_prompts)

                    # Check all remaining prompts if they are "neighbors"
                    for other_p in remaining_list:
                        # If the IoU is above the threshold, treat them as the same object
                        # i.e. they are a "neighbor" for BFS clustering
                        iou_val = bounding_box_iou(p.box, other_p.box)
                        if iou_val > 0.4 and abs(p.z - other_p.z) <= z_dist_max:
                            other_p.obj_id = current_obj_id
                            remaining_prompts.remove(other_p)
                            queue.append(other_p)

                current_obj_id += 1


        # --- Step 3: Group prompts by assigned object id ---
        prompt_by_obj_id = defaultdict(list)
        for prompts_for_single_class in prompts_by_class.values():
            for p in prompts_for_single_class:
                prompt_by_obj_id[p.obj_id].append(p)


        # --- Step 4: Select top N prompts per object, spaced in z ---
        final_prompts = []
        for prompts_for_single_class in prompt_by_obj_id.values():
            p_sorted = sorted(prompts_for_single_class, key=lambda p: p.quality_score, reverse=True)
            selected = []

            while p_sorted and len(selected) < n_prompts_per_obj:
                best = p_sorted.pop(0)
                selected.append(best)
                # Remove prompts too close in z
                p_sorted = [p for p in p_sorted if abs(p.z - best.z) >= multiprompt_z_spacing]

            final_prompts.extend(selected)

        print(f"Generated {len(final_prompts)} prompts across {len(prompt_by_obj_id)} objects.")
        return final_prompts


    def __box_quality__(self, bboxes, scores):
        """
        Calculate the quality of a bounding box based on its size and confidence score.
        Args:
            bboxes (numpy.ndarray): Array of bounding boxes in the format [x1, y1, x2, y2].
            scores (numpy.ndarray): Confidence scores for each bounding box.
        Returns:
            numpy.ndarray: Array of quality scores for each bounding box.
        """
        if bboxes.shape[0] == 0:
            return np.array([])

        # Calculate the area of each bounding box
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        areas = widths * heights

        # Calculate the quality score as the product of area and confidence score
        quality_scores = np.log(areas) * scores**2

        return quality_scores