import torch

"""
These classes should be used for other BOB models to ensure they follow the YOLO-style interface.
"""


class YOLOResult:
    """Simple class to mimic YOLO result structure"""

    def __init__(self, xyxy, conf, cls):
        self.boxes = SimpleNamespace()
        self.boxes.xyxy = xyxy
        self.boxes.conf = conf
        self.boxes.cls = cls


class SimpleNamespace:
    """Simple namespace class"""

    def __init__(self):
        pass


class YOLOStylePredictor:
    def __init__(self, model):
        self.model = model
        self.names = {}

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
        raise NotImplementedError(
            "This method should be implemented in a subclass. "
        )
