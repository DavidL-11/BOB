import os
import onnxruntime as ort
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from BOB.inference.yolo_style_predictor import (
    YOLOResult,
    YOLOStylePredictor,
)
import time


class DFinePredictor(YOLOStylePredictor):
    """
    Copied from D-FINE/tools/inference/torch_inf.py and adjusted to fit the YOLO-style interface.
    """

    def __init__(self, checkpoint_path=None):
        file_location = os.path.dirname(os.path.abspath(__file__))
        # Get the BOB root directory
        root_dir = os.path.abspath(os.path.join(file_location, ".."))

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                root_dir, "checkpoints", "D-FINE-N", "D-FINE-N.onnx"
            )

        self.model = ort.InferenceSession(checkpoint_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print(f"Using device: {ort.get_device()}")

        # Add model.names property to match YOLO interface
        self.names = {
            0: "Glottis",
            1: "Left Vocal Cord",
            2: "Right Vocal Cord",
            3: "Lung",
            4: "Polyp",
            5: "Tool",
            6: "Organ",
            7: "Liver",
            8: "Right Kidney",
            9: "Spleen",
            10: "Mitochondria",
            11: "Aorta",
            12: "Inferior Vena Cava",
            13: "Pharynx",
            14: "Fetal Head",
            15: "Gallbladder",
            16: "Esophagus",
            17: "Stomach",
            18: "Tooth",
            19: "Left Kidney",
            20: "Prostate/Uterus",
            21: "Skin Lesion",
            22: "Glioma",
            23: "Optic Disc",
            24: "Optic Cup",
            25: "Nucleus",
            26: "Heart Myocardium",
            27: "Heart Left Ventricle",
            28: "Heart Right Ventricle",
            29: "Heart Atrium Left",
        }

    def draw(self, images, labels, boxes, scores):
        for i, im in enumerate(images):
            draw = ImageDraw.Draw(im)

            scr = scores[i]
            lab = labels[i]
            box = boxes[i]

            for j, b in enumerate(box):
                draw.rectangle(list(b), outline="red")
                draw.text(
                    (b[0], b[1]),
                    text=f"{lab[j].item()} {round(scr[j].item(), 2)}",
                    fill="blue",
                )

            import matplotlib.pyplot as plt

            plt.imshow(im)
            plt.axis("off")
            plt.show()

    def resize_with_aspect_ratio(self, image, size, interpolation=Image.BILINEAR):
        """Resizes an image while maintaining aspect ratio and pads it."""
        original_width, original_height = image.size
        ratio = min(size / original_width, size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        image = image.resize((new_width, new_height), interpolation)

        # Create a new image with the desired size and paste the resized image onto it
        new_image = Image.new("RGB", (size, size))
        new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
        return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2

    def process_image(
        self,
        model,
        img,
        conf_threshold=0.5,
        max_det=100,
        classes=None,
    ):
        resized_im_pil, ratio, pad_w, pad_h = self.resize_with_aspect_ratio(img, 512)
        orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

        transforms = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        im_data = transforms(resized_im_pil).unsqueeze(0)
        
        output = model.run(
            output_names=None,
            input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
        )

        labels, boxes, scores = output

        # Convert to YOLO-style output format with confidence filtering
        if len(labels) > 0 and len(labels[0]) > 0:
            # Extract data from first batch element (already numpy arrays from ONNX)
            bboxes = boxes[0]  # Shape: [N, 4] in xyxy format
            scores_array = scores[0]  # Shape: [N]
            class_ids = labels[0].astype(int)  # Shape: [N]

            # Apply confidence filtering
            conf_mask = scores_array >= conf_threshold
            filtered_bboxes = bboxes[conf_mask]
            filtered_scores = scores_array[conf_mask]
            filtered_class_ids = class_ids[conf_mask]

            # Class filtering if classes is provided
            if classes is not None:
                class_mask = np.isin(filtered_class_ids, classes)
                filtered_bboxes = filtered_bboxes[class_mask]
                filtered_scores = filtered_scores[class_mask]
                filtered_class_ids = filtered_class_ids[class_mask]

            # Apply max_det filtering - keep only the highest scoring detections
            if len(filtered_scores) > max_det:
                # Get indices of top max_det scores
                top_indices = np.argsort(filtered_scores)[-max_det:]
                filtered_bboxes = filtered_bboxes[top_indices]
                filtered_scores = filtered_scores[top_indices]
                filtered_class_ids = filtered_class_ids[top_indices]

            # Adjust bounding boxes according to the resizing and padding
            filtered_bboxes = [
                [
                    (bb[0] - pad_w) / ratio,
                    (bb[1] - pad_h) / ratio,
                    (bb[2] - pad_w) / ratio,
                    (bb[3] - pad_h) / ratio,
                ]
                for bb in filtered_bboxes
            ]

            # Create YOLO-style result object with filtered data
            result = YOLOResult(
                xyxy=torch.tensor(filtered_bboxes),
                conf=torch.tensor(filtered_scores),
                cls=torch.tensor(filtered_class_ids),
            )
        else:
            # No detections found
            result = YOLOResult(
                xyxy=torch.tensor([]).reshape(0, 4),
                conf=torch.tensor([]),
                cls=torch.tensor([]),
            )

        return [result]

    def predict(
        self,
        source,
        imgsz=512,
        conf=0.5,
        iou=0.7,
        max_det=100,
        show=False,
        classes=None,
        device=None,
        verbose=False,
    ):
        """
        The predict method processes an image or video source and returns detection results.
        It follows the YOLO-style interface, accepting arguments and returning results in a format compatible with the ultralytics YOLO implementation.
        Note that since it is D-FINE-based, it doesn't need NMS (non-maximum suppression),
        so the iou parameter is not used here.
        """
        # Convert NP array to PIL Image
        if isinstance(source, np.ndarray):
            source = Image.fromarray(source)
        if isinstance(source, Image.Image):
            source = source.convert("RGB")

        # Process as image - now returns YOLO-style results with confidence filtering applied
        results = self.process_image(
            model=self.model,
            img=source,
            conf_threshold=conf,
            max_det=max_det,
            classes=classes,
        )


        if show and len(results) > 0 and len(results[0].boxes.conf) > 0:
            # Extract data for visualization using YOLO-style access
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy().astype(int)

            # Convert to format expected by draw function
            labels_tensor = [torch.tensor(labels)]
            boxes_tensor = [torch.tensor(boxes)]
            scores_tensor = [torch.tensor(scores)]

            self.draw([source], labels_tensor, boxes_tensor, scores_tensor)

        return results


if __name__ == "__main__":
    img_path = "/home/david/Documents/segFM/Datasets/BAGLS/test/0.png"
    img = Image.open(img_path).convert("RGB")

    predictor = DFinePredictor()
    predictor.predict(img, show=True)
