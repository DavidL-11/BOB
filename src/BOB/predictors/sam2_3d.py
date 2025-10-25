from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2_video_predictor

from BOB.checkpoints import checkpoints
from BOB.inference import utils

torch.set_float32_matmul_precision("high")

class SAM2Predictor3D:
    """
    A class to handle the SAM2 model for 3D segmentation tasks.
    This model is different compared to MedSAM2 in the sense that it is a true 3D segmentation foundation model.
    It initializes the model and provides methods to segment images.
    """

    def __init__(self, checkpoint=checkpoints.SAM2_tiny, config=checkpoints.SAM2_tiny_cfg):
        self.checkpoint = checkpoint
        self.model_cfg = config
        self.device = utils.setup_device()
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=self.device)
        np.random.seed(11)

    def resize_grayscale_to_rgb(self, array):
        """
        Resize a 3D grayscale NumPy array to an RGB image.

        Parameters:
            array (np.ndarray): Input array of shape (d, h, w).

        Returns:
            np.ndarray: Resized array of shape (d, 3, image_size, image_size).
        """
        d, h, w = array.shape
        resized_array = np.zeros((d, 3, h, w), dtype=np.uint8)

        for i in range(d):
            img_pil = Image.fromarray(array[i].astype(np.uint8))
            img_rgb = img_pil.convert("RGB")
            img_array = np.array(img_rgb).transpose(
                2, 0, 1
            )  # (3, image_size, image_size)
            resized_array[i] = img_array

        return resized_array


    def get_largest_bounding_boxes(self, mask, plot_prompt=False):
        """
        Returns as many bounding boxes as there are objects in the mask.
        Format is z, List([x1, y1, x2, y2])
        The mask is expected to be a 3D numpy array.
        The bounding box is a 2D array obtained by slicing the mask along the z-axis at a good coordinate
        and then finding the bounding box of the ground truth.
        """
        object_colors = np.unique(mask)  # Retrieve all the unique colors in the mask
        # Remove the background color (0)
        object_colors = object_colors[object_colors > 0]

        boxes = {}

        # Iterate over the unique colors and find the bounding box for each color
        for color in object_colors:
            color_mask = (mask == color).astype(np.uint8)

            # Find the slice with the most pixels
            z = np.argmax(np.sum(color_mask, axis=(1, 2)))
            # Find the bounding box of the color mask
            x1, y1, w, h = cv2.boundingRect(color_mask[z])
            # Add it to the dictionary
            boxes[color] = (z, np.array([x1, y1, x1 + w, y1 + h]))

        if plot_prompt:  # Plot the bounding box
            for color in object_colors:
                z, (x1, y1, x2, y2) = boxes[color]
                rect = cv2.rectangle(mask[z].copy(), (x1, y1), (x2, y2), (255, 0, 0), 2)
                plt.imshow(rect, cmap="gray")
                plt.show()
        return boxes


    def convert_3d_image_to_frames(self, image, folder):
        """
        Convert a 3D image to a list of 2D frames and save them in a given directory.
        Args:
            image (np.ndarray): 3D image of shape (depth, rgb_channels, height, width).
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Check if the image is grayscale and convert it to RGB + resize
        image = self.resize_grayscale_to_rgb(image)

        depth = image.shape[0]

        for i in range(depth):
            frame = np.transpose(
                image[i], (1, 2, 0)
            )  # Convert to (height, width, channels)
            # Convert the frame to numpy array from torch tensor and save it
            cv2.imwrite(os.path.join(folder, f"{i:03d}.jpg"), frame)

    def predict(self, image, prompts):
        """
        Segments a 3D image using the SAM2 model and evaluates the segmentation results.
        Args:
            data (dict): Dictionary containing the image, ground truth, and prompts.
            dataset (Dataset): The dataset object containing metadata and utility functions.
        Returns:
            pd.DataFrame: A DataFrame containing the segmentation metrics for each object.
        """
        width, height, depth = image.shape[::-1]

        folder = "temp_frames"

        self.convert_3d_image_to_frames(image, folder)

        with torch.inference_mode():
            inference_state = self.predictor.init_state(video_path="temp_frames")

            prompts_by_obj = defaultdict(list)
            for prompt in prompts:
                prompts_by_obj[prompt.obj_id].append(prompt)

            # Create n_obj result arrays, one for each object
            result_arrays = {obj_id: np.zeros((depth, height, width), dtype=np.uint8) for obj_id in prompts_by_obj.keys()}
            result_names = {obj_id: f"{prs[0].class_label} {obj_id}" for obj_id, prs in prompts_by_obj.items()}

            for obj_id, prompts in prompts_by_obj.items():
                # Add all prompts with the same object ID to the predictor
                for prompt in prompts:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate forward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state):
                # Set the result array to 1 where the mask logits are greater than 0
                    result_arrays[obj_id][out_frame_idx, (out_mask_logits[0] > 0.5).cpu().numpy()[0]] = obj_id

                # Reset the predictor state
                self.predictor.reset_state(inference_state)
                
                # Add all prompts with the same object ID to the predictor
                for prompt in prompts:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate backward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state, reverse=True):
                    # Save results in the result array
                    result_arrays[obj_id][out_frame_idx, (out_mask_logits[0] > 0.5).cpu().numpy()[0]] = obj_id

                self.predictor.reset_state(inference_state)

        # Remove all files in the temporary directory
        for image in os.listdir(folder):
            image_path = os.path.join(folder, image)
            if os.path.isfile(image_path):
                os.remove(image_path)
        os.rmdir(folder)

        return result_arrays, result_names