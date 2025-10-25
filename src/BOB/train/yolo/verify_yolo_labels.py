import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

"""
This script visualizes YOLO labels by plotting them on random images from a specified dataset.
This is useful for verifying the correctness of the YOLO label format
and ensuring that bounding boxes are correctly placed.
"""

# Set paths
base_dir = 'BOB_dataset/ToothFairy3/train'  # Adjust this path as needed
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

# Get list of image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
sample_images = random.sample(image_files, min(50, len(image_files)))  # Display up to 5 random images

def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes  # No labels for this image
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed lines
            cls_id, x_center, y_center, width, height = map(float, parts)
            boxes.append((int(cls_id), x_center, y_center, width, height))
    return boxes

# Plot images with bounding boxes
for img_file in sample_images:
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

    # Load image
    image = Image.open(img_path).convert('RGB')
    w, h = image.size

    # Load labels
    boxes = load_yolo_labels(label_path)

    # Plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for cls_id, x_center, y_center, width, height in boxes:
        # Convert YOLO to corner box format
        x_min = (x_center - width / 2) * w
        y_min = (y_center - height / 2) * h
        box_w = width * w
        box_h = height * h

        area_rel = width * height * 100  # Area relative to image size in percentage

        # Calculate the mean intensity in the box
        img_patch=image.crop((x_min, y_min, x_min + box_w, y_min + box_h))
        mean_intensity = sum(img_patch.convert("L").getdata()) / (box_w * box_h)
        #print(f"Box class {cls_id} mean intensity: {mean_intensity:.2f}")

        # Draw box
        rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, str(cls_id) + f" ({area_rel:.2f}%) {mean_intensity:.1f}", color='yellow', fontsize=12, weight='bold')

    plt.title(f"Image: {img_file} with {len(boxes)} boxes")
    plt.axis('off')
    plt.show()
