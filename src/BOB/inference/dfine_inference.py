import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

from BOB.train.dfine.model.src.core import YAMLConfig
from BOB.inference.yolo_style_predictor import (
    YOLOResult,
    YOLOStylePredictor,
)


class DFinePredictor(YOLOStylePredictor):
    """
    Copied from D-FINE/tools/inference/torch_inf.py and adjusted to fit the YOLO-style interface.
    """

    def __init__(self, checkpoint_path=None, config_path=None):
        file_location = os.path.dirname(os.path.abspath(__file__))
        # Get the BOB root directory
        root_dir = os.path.abspath(os.path.join(file_location, ".."))

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                root_dir, "checkpoints", "D-FINE-N", "D-FINE-N.pt"
            )
        if config_path is None:
            config_path = os.path.join(
                root_dir, "train", "dfine", "dfine_hgnetv2_n_custom.yml"
            )
        """Main function"""
        cfg = YAMLConfig(cfg_path=config_path, resume=checkpoint_path)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Use EMA weights if included in the large checkpoint (The one that contains optimizer, etc.)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        # Use model weights if no EMA weights are available (Still in the large checkpoint format)
        elif "model" in checkpoint:
            state = checkpoint["model"]
        # Otherwise, assume checkpoint is already a state_dict
        else:
            state = checkpoint

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.model = Model().to("cuda")

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

    def process_image(
        self,
        model,
        device,
        img,
        conf_threshold=0.5,
        max_det=100,
        img_size=512,
        classes=None,
    ):
        w, h = img.size
        orig_size = torch.tensor([[w, h]]).to(device)

        transforms = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ]
        )
        im_data = transforms(img).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Convert to YOLO-style output format with confidence filtering
        if len(labels) > 0 and len(labels[0]) > 0:
            # Extract data from first batch element and convert to numpy
            bboxes = boxes[0].detach().cpu().numpy()  # Shape: [N, 4] in xyxy format
            scores_array = scores[0].detach().cpu().numpy()  # Shape: [N]
            class_ids = labels[0].detach().cpu().numpy().astype(int)  # Shape: [N]

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
        device=("cuda" if torch.cuda.is_available() else "cpu"),
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
        elif isinstance(source, Image.Image):
            source = source.convert("RGB")

        # Process as image - now returns YOLO-style results with confidence filtering applied
        results = self.process_image(
            self.model,
            device=device,
            img=source,
            conf_threshold=conf,
            max_det=max_det,
            img_size=imgsz,
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
    img = Image.open(img_path).convert("RGB").resize((512, 512))

    predictor = DFinePredictor()
    predictor.predict(img, show=True)
