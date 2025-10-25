import os
from ultralytics import YOLO

model = YOLO("yolo12n.pt")  # Load a pretrained YOLO model

results = model.train(
    data=os.path.join(os.path.dirname(__file__), "dataset.yaml"),  # Path to dataset configuration
    imgsz=512,  # Image size for training
    epochs=50,
    batch=80,
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    name="BOB_Y12n",  # Name of the training run
    save=True,  # Save the model after training
    device="0",  # Specify the GPU device to use
    resume=False,  # Resume training from the last checkpoint
    workers=8,  # Number of worker threads for data loading
    flipud=0.2,  # Probability of flipping images upside down
    fliplr=0.3,  # Probability of flipping images left to right
    optimizer="AdamW",  # Optimizer to use
    plots=True,  # Generate training plots
)