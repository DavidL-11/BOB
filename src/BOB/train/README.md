## Creating the combined dataset
To create the combined dataset, you need to follow these steps:

1. Prepare a DataLoader for each of the datasets you want to combine. Ensure that each DataLoader is properly configured with box prompts.
2. Add the DataLoaders to the convert_datasets.py script
3. Run the convert_datasets.py script to generate the combined dataset. It will create a new directory containing 'n_samples' images from each dataset, split into training and validation sets. It will also generate corresponding YOLO-format '.txt' annotation files for each image. The 'combine_datasets' method at the end will combine all subfolders into a single, combined folder with train and val subfolders. Object Detection models should be trained on this combined folder.

## Training the object detection model
To train the object detection model using the combined dataset, use the corresponding training script (e.g., train_yolo.py) in the BOB/train directory. 