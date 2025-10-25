import os
import json
import glob
import yaml
from PIL import Image


def load_categories_from_yaml(yaml_path):
    """
    Parse dataset.yaml and return categories dict {id: name}.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "names" not in data:
        raise ValueError("YAML file must contain a 'names' field.")

    categories = {}
    # names can be a list or dict depending on YOLO version
    if isinstance(data["names"], list):
        for i, name in enumerate(data["names"]):
            categories[i] = name
    elif isinstance(data["names"], dict):
        categories = {int(k): v for k, v in data["names"].items()}
    else:
        raise ValueError("'names' in YAML must be a list or dict.")

    return categories


def yolo_to_coco(yolo_folder, image_folder, output_json, categories):
    """
    Convert YOLO format labels to COCO format.
    """

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Build category list
    for cid, name in categories.items():
        coco["categories"].append({
            "id": cid,
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1

    for txt_file in glob.glob(os.path.join(yolo_folder, "*.txt")):
        # Match image file
        base = os.path.splitext(os.path.basename(txt_file))[0]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            test_path = os.path.join(image_folder, base + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break
        if img_path is None:
            print(f"Warning: No image found for {txt_file}, skipping.")
            continue

        # Get image size
        with Image.open(img_path) as im:
            w, h = im.size

        # Add image entry
        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h
        })

        # Parse YOLO annotations
        with open(txt_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x, y, bw, bh = parts
                cls_id = int(cls_id)
                x, y, bw, bh = map(float, (x, y, bw, bh))

                # Convert YOLO -> COCO
                x_min = (x - bw / 2) * w
                y_min = (y - bh / 2) * h
                box_w = bw * w
                box_h = bh * h

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"COCO annotations saved to {output_json}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "..", "yolo", "dataset.yaml")  # same folder as script

    for category in ["train", "val"]:
        output_dir = os.path.join("BOB_dataset_combined", category, "annotations")
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        yolo_folder = f"BOB_dataset_combined/{category}/labels"       # folder with YOLO txt files
        image_folder = f"BOB_dataset_combined/{category}/images"      # folder with corresponding images
        output_json = os.path.join(output_dir, f"{category}.json")

        # Load categories dynamically
        categories = load_categories_from_yaml(yaml_path)

        yolo_to_coco(yolo_folder, image_folder, output_json, categories)
