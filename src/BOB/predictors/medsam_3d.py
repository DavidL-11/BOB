import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from sam2.build_sam import build_sam2_video_predictor_npz

from BOB.checkpoints import checkpoints

from BOB.logger import logger
from BOB.inference import utils

torch.set_float32_matmul_precision("high")

class MedSAM2Predictor3D:
    """
    A class to handle the MedSAM2 model for 3D segmentation tasks.
    It initializes the model and provides methods to segment images.
    It is a stripped-down version of the segFM/predictors/medsam_3d.py file, adapted for BOB.
    """

    def __init__(self, checkpoint=checkpoints.MedSAM2_CT):
        self.checkpoint = checkpoint
        self.model_cfg = checkpoints.MedSAM_cfg
        self.device = utils.setup_device()

        self.predictor = build_sam2_video_predictor_npz(checkpoints.MedSAM_cfg, checkpoint, device=self.device)

    def predict(self, image, prompts):
        """
        Segment the image using the MedSAM2 model.
        Args:
            image (np.ndarray): The input image in (z, y, x, rgb) format.
            prompts (list): List of prompts to use for segmentation.
        Returns:
            tuple: A tuple containing the segmented result arrays and the object names, identified by object IDs.
        """

        # Image is in (z, y, x, rgb)
        depth, height, width = image.shape[0], image.shape[1], image.shape[2]

        image = utils.resize_and_normalize(image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(image, height, width)

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
                color = prompts[0].color if prompts[0].color else 1  # Default color if not specified
                label = prompts[0].class_label

                self.predictor.reset_state(inference_state)

                for prompt in prompts:
                    logger.info(
                        f"Adding prompt for {label} at frame {prompt.z} with color {prompt.color}"
                    )
                    # Add the prompts to the predictor
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
                    result_arrays[obj_id][out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = obj_id

                # Reset the predictor state
                self.predictor.reset_state(inference_state)

                # Add the prompts again for backward propagation
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
                    # Set the result array to 1 where the mask logits are greater than 0
                    result_arrays[obj_id][out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = obj_id

                self.predictor.reset_state(inference_state)

        logger.info("Segmentation completed.")

        return result_arrays, result_names