import numpy as np
import os
import pandas as pd
import shutil
import cv2
import random

from segFM.DataLoaders import brats_medical_decathlon, flare22, amos22, toothfairy3
from segFM.logger import logger
from segFM.DataLoaders.base_dataset import BaseImageDataset, BaseVideoDataset
from segFM import utils, prompts

class DatasetConverter():
    """
    DatasetConverter is a class that provides methods to convert various segmentation datasets into a format
    suitable for object detection tasks. For each dataset, it splits the data into training and validation sets,
    performs transformations and data augmentation (e.g. rotation, noise addition), generates bounding boxes 
    from the ground truth segmentation masks, converts those bounding boxes into the YOLO format,
    and saves the results in a structured directory.
    Video datasets get converted to images first, and then processed similarly to image datasets.

    Folder structure:
    ```
    output_directory/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

    Labels are saved in the YOLO format, which specifies the class index, center x, center y, width, and height of the bounding box,
    normalized to 0-1 range based on the image dimensions.
    """
    def __init__(self, dataset: BaseImageDataset | BaseVideoDataset, base_path: str, allow_empty: bool = False):
        """
        Initialize the DatasetConverter with the specified dataset and output directory.
        
        Args:
            dataset (BaseDataset): An instance of a dataset class that inherits from BaseDataset.
            output_directory (str): The directory where the converted dataset will be saved.
        """
        logger.debug(f"Initializing DatasetConverter for {dataset.__class__.__name__} dataset.")

        self.dataset = dataset
        if dataset is None:
            return

        self.output_directory = base_path + f"/{dataset.name}"
        self.allow_empty = allow_empty

        # Actually not even needed since YOLO trainer already does data augmentation
        #self.data_augmenter = DataAugmenter()
        self.create_directories(self.output_directory)

        logger.info(f"Output directories created at {self.output_directory}.")

    def create_directories(self, output_directory: str):
        """
        Create the necessary output directories for training and validation sets.
        Args:
            output_directory (str): The base directory where the dataset will be saved.
        """
        # Create output directories for training and validation sets
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(os.path.join(output_directory, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_directory, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_directory, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_directory, 'val', 'labels'), exist_ok=True)

    def convert(self, n_images=-1):
        """
        Convert the dataset into a format suitable for object detection tasks.
        This method checks the type of dataset (image or video) and calls the appropriate conversion method.
        Raises:
            ValueError: If the dataset type is not supported (not a subclass of BaseImageDataset or BaseVideoDataset).
        """
        # Check if the dataset is a video dataset or an image dataset
        if isinstance(self.dataset, BaseVideoDataset):
            logger.info("Converting video dataset.")
            self.__convert_videos__()
        elif isinstance(self.dataset, BaseImageDataset):
            if isinstance(self.dataset, flare22.Flare22) or isinstance(self.dataset, amos22.Amos22):
                logger.info(f"Converting 3D images from the dataset {self.dataset.__class__.__name__}.")
                self.__convert_3d_images__(dataset=self.dataset, n_images=n_images, slice_amount=0.1)
            elif isinstance(self.dataset, toothfairy3.ToothFairy3):
                logger.info(f"Converting 3D images from the dataset {self.dataset.__class__.__name__}.")
                self.__convert_3d_images__(dataset=self.dataset, n_images=n_images, slice_amount=0.1)
            elif isinstance(self.dataset, brats_medical_decathlon.BratsMSDGlioma):
                logger.info(f"Converting 3D images from the dataset {self.dataset.__class__.__name__}.")
                self.__convert_3d_images__(dataset=self.dataset, n_images=n_images, slice_amount=0.1)
            else:
                logger.info("Converting 2D images from the dataset.")
                self.__convert_images__(n_images=n_images)
        else:
            raise ValueError("Unsupported dataset type. Must be a subclass of BaseImageDataset or BaseVideoDataset.")

    def __convert_images__(self, n_images=-1):
        if n_images == -1:
            n_images = self.dataset.n_images
        else:
            n_images = min(n_images, self.dataset.n_images)
            
        # Get all images from the dataset
        indices = self.dataset.get_n_nonduplicate_indeces(n_images)

        # Shuffle the indeces and split into training and validation sets
        train, val = utils.split_dataset_indeces(indices, 0.8)

        logger.debug(f"Number of images to convert: {len(indices)}")

        # Iterate over both splits
        for split in ['train', 'val']:
            # Select the appropriate indices for the current split
            split_indeces = train if split == 'train' else val

            logger.info(f"Converting {split} dataset with {len(split_indeces)} images.")
            # Iterate over the indices for the current split
            for idx, i in enumerate(split_indeces):
                # Get the data dictionary for the current index
                data = self.dataset[i] 

                if data is None:
                    logger.warning(f"Data for index {i} is None. Skipping this index.")
                    continue

                # Print a progress bar in the console for visual feedback
                utils.printProgressBar(idx + 1, len(split_indeces), prefix=f'[{split}] Progress:', suffix='Complete', length=70)
                img = data['img']
                name = data['name']

                # Generate the yolo label from the bboxes stored in the data dictionary
                label = utils.generate_yolo_label(data)
                

                # Skip the image with a 75% chance if no objects are detected
                if label == "" and random.random() > 0.25:
                    continue #Keep some images without objects for better training

                img_path = os.path.join(self.output_directory, split, 'images', name)
                label_path = os.path.join(self.output_directory, split, 'labels', f"{name.split('.')[0]}.txt")

                # Ensure the image is in the correct format (BGR for OpenCV)
                if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Save the image and label in the appropriate directories
                cv2.imwrite(img_path, img)
                with open(label_path, 'w') as f:
                    f.write(label)

    def __convert_3d_images__(self, dataset, n_images=-1, slice_amount=0.3):
        """
        Convert 3D images from the dataset into a format suitable for object detection tasks.

        Args:
            dataset (BaseImageDataset): The dataset containing 3D images.
            slice_amount (float): The fraction of slices to randomly select from each 3D image for conversion. 
                                  0.1 corresponds to 10% of the slices.
        """
        if n_images == -1:
            n_images = self.dataset.n_images
        else:
            n_images = min(n_images, self.dataset.n_images)

        # Get all images from the dataset
        indices = self.dataset.get_n_nonduplicate_indeces(n_images)

        # Shuffle the indeces and split into training and validation sets
        train, val = utils.split_dataset_indeces(indices, 0.8)

        logger.debug(f"Number of images to convert: {len(indices)}")

        # Iterate over both splits
        for split in ['train', 'val']:
            # Select the appropriate indices for the current split
            split_indeces = train if split == 'train' else val

            logger.info(f"Converting {split} dataset with {len(split_indeces)} images.")
            # Iterate over the indices for the current split
            for idx, i in enumerate(split_indeces):
                # Get the data dictionary for the current index
                data = self.dataset[i] 

                if data is None:
                    logger.warning(f"Data for index {i} is None. Skipping this index.")
                    continue

                # Print a progress bar in the console for visual feedback
                utils.printProgressBar(idx + 1, len(split_indeces), prefix=f'[{split}] Progress:', suffix='Complete', length=70)
                img = data['img'] # Shape: z, w, w
                gt = data['gt'] # Ground truth segmentation mask
                name = data['name']

                # Iterate over the slices of the 3D image
                for z in range(img.shape[0]):
                    # Extract the 2D slice
                    img_slice = img[z, :, :]
                    gt_slice = gt[z, :, :]

                    # Only use 30% of the slices
                    if random.random() > slice_amount:
                        # Check if the "perfect" prompts contain this slice, then keep it - else discard
                        exit = True
                        for pr in data['prompts']:
                            if pr.z == z:
                                exit = False

                        if exit:
                            continue

                    prompt = prompts.get_multicolor_prompt(gt=gt_slice, image=img_slice, dataset=dataset, mode="box")

                    # Update the data dictionary with the current slice and its bounding boxes
                    data['img'] = img_slice
                    data['prompts'] = prompt

                    # Generate the yolo label from the bboxes stored in the data dictionary
                    label = utils.generate_yolo_label(data, min_area=0.002)

                    # Keep some images without objects for better training
                    if (label == "" or label is None) and random.random() > 0.1:
                        logger.debug(f"No objects detected in slice {z} of {name}. Skipping this slice.")
                        continue

                    # Give each slice a unique name based on the original image name and the z index
                    slice_name = f"{name.split('.')[0]}_z{z}"

                    img_path = os.path.join(self.output_directory, split, 'images', f"{slice_name}.png")
                    label_path = os.path.join(self.output_directory, split, 'labels', f"{slice_name}.txt")

                    # Ensure the image is in the correct format (BGR for OpenCV)
                    if img_slice.ndim == 2 or (img_slice.ndim == 3 and img_slice.shape[2] == 1):
                        img_slice = cv2.cvtColor(img_slice, cv2.COLOR_GRAY2BGR)
                    else:
                        img_slice = cv2.cvtColor(img_slice, cv2.COLOR_RGB2BGR)

                    # Save the image and label in the appropriate directories
                    cv2.imwrite(img_path, img_slice)
                    with open(label_path, 'w') as f:
                        f.write(label)

    def __convert_videos__(self):
        # Get all videos from the dataset
        data = self.dataset.get_n_nonduplicate_indeces(self.dataset.n_videos)

        logger.debug(f"Number of videos to convert: {len(data)}")

    def combine_datasets(self, base_path):
        """
        Combine multiple datasets into a single dataset. This is required for training a model on multiple datasets,
        but keeping the datasets separate is more understandable and easier to manage.

        Folder structure before combining:
        ```
        base_path/
        ├── dataset1/
        │   ├── train/
        │   │   ├── images/
        │   │   └── labels/
        │   └── val/
        │       ├── ...
        ├── dataset2/
        │   ├── train/
        │   │   ├── ..
        │   └── val/
        │       ├── ...
        └── dataset3/
            ├── train/
            │   ├── images/
            │   └── labels/
            └── val/
                ├── images/
                └── labels/
        ```

        Folder structure after combining:

        ```
        base_path/
        └── combined/
            ├── train/
            └── val/
        ```

        Note:
            The original folder structure of the datasets is preserved.

        Args:
            base_path (str): The base path where the datasets are located.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The base path {base_path} does not exist.")
        elif not os.path.isdir(base_path):
            raise NotADirectoryError(f"The base path {base_path} is not a directory.")
        elif not os.listdir(base_path):
            raise ValueError(f"The base path {base_path} is empty. No datasets to combine.")

        logger.info(f"Combining datasets from {base_path} into a single dataset.")
        logger.info(f"Found {len(os.listdir(base_path))} datasets to combine.")

        combined_path = base_path +'_combined'
        os.makedirs(combined_path, exist_ok=True)
        os.makedirs(os.path.join(combined_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(combined_path, 'val'), exist_ok=True)

        # Iterate through each dataset directory
        for dataset_name in os.listdir(base_path):
            dataset_path = os.path.join(base_path, dataset_name)

            if os.path.isdir(dataset_path):
                logger.info(f"Processing dataset: {dataset_name}")

                train_path = os.path.join(dataset_path, 'train')
                val_path = os.path.join(dataset_path, 'val')

                if os.path.exists(train_path):
                    # Copy images and labels from the train directory
                    shutil.copytree(os.path.join(train_path, 'images'), os.path.join(combined_path, 'train', 'images'), dirs_exist_ok=True)
                    shutil.copytree(os.path.join(train_path, 'labels'), os.path.join(combined_path, 'train', 'labels'), dirs_exist_ok=True)
                    logger.info(f"Copied training data from {dataset_name}.")

                if os.path.exists(val_path):
                    # Copy images and labels from the val directory
                    shutil.copytree(os.path.join(val_path, 'images'), os.path.join(combined_path, 'val', 'images'), dirs_exist_ok=True)
                    shutil.copytree(os.path.join(val_path, 'labels'), os.path.join(combined_path, 'val', 'labels'), dirs_exist_ok=True)
                    logger.info(f"Copied validation data from {dataset_name}.")
