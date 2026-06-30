import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict
import time
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from BOB.logger import logger

torch.set_float32_matmul_precision("high")

class nnInteractivePredictor3D:
    """
    A class to handle the nnInteractive model for 3D segmentation tasks.
    It initializes the model and provides methods to segment images.
    """

    def __init__(self, model_name="nnInteractive_v1.0"):
        REPO_ID = "nnInteractive/nnInteractive"
        DOWNLOAD_DIR = "."  # Specify the download directory
        self.model_name = model_name
        
        download_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{self.model_name}/*"],
            local_dir=DOWNLOAD_DIR
        )
        
        self.session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        
        model_path = os.path.join(DOWNLOAD_DIR, self.model_name)
        self.session.initialize_from_trained_model_folder(model_path)
        

    def predict(self, image: np.ndarray, prompts: list):
        """
        Segment the image using the MedSAM2 model.
        Args:
            image (np.ndarray): The input image in (z, y, x) format.
            prompts (list): List of prompts to use for segmentation.
        Returns:
            tuple: A tuple containing the segmented result arrays and the object names, identified by object IDs.
        """
        if image.ndim != 3:
            raise ValueError("nnInteractive3D only supports 3D grayscale images!")

        depth, height, width = image.shape[0], image.shape[1], image.shape[2]
        image = np.transpose(image, (2, 1, 0))[None, ...]
        
        # Set the image and prepare the target buffer (-> result array)
        self.session.set_image(image)
        target_tensor = torch.zeros(image.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(target_tensor)
        
        # Cluster prompts by object ID -> Returns 2D list of prompts
        # Each sublist contains prompts for a specific object
        # Example: [[prompt1, prompt2], [prompt3], ...]
        prompts_by_obj = defaultdict(list)
        for prompt in prompts:
            prompts_by_obj[prompt.obj_id].append(prompt)

        # Create n_obj result arrays, one for each object
        result_arrays = {obj_id: np.zeros((depth, height, width), dtype=np.uint8) for obj_id in prompts_by_obj.keys()}
        result_names = {obj_id: f"{prs[0].class_label} {obj_id}" for obj_id, prs in prompts_by_obj.items()}

        for obj_id, prompts in prompts_by_obj.items():
            # Reset the result array for each prompt
            label = prompts[0].class_label

            for prompt in prompts:
                box = prompt.box # (4, ) in xyxy format
                z = prompt.z # Scalar value indicating the slice index
                
                # ATTENTION: nnInteractive has different bbox format than SAM
                # BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
                # Note: nnInteractive pre-trained models currently only support **2D bounding boxes**.
                # This means that **one dimension must be [d, d+1]** to indicate a single slice.
                box = np.array([
                    [box[0], box[2]],  # x1, x2
                    [box[1], box[3]],  # y1, y2
                    [z, z + 1]         # z1, z2
                ])
                
                logger.info(
                    f"Adding prompt for {label} at frame {prompt.z} with color {prompt.color}"
                )
                
                # Add the prompts to the predictor
                self.session.add_bbox_interaction(
                    bbox_coords=box,
                    include_interaction=True,
                )
                

            results = target_tensor.clone()
            # Transpose results from (x, y, z) back to (z, y, x) to match result_arrays format
            results_transposed = results.numpy().transpose(2, 1, 0)
            result_arrays[obj_id][results_transposed > 0] = obj_id
            # Remove the boxes to prepare for the next object
            self.session.reset_interactions()

        logger.info("Segmentation completed.")

        return result_arrays, result_names