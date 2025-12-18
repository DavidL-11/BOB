import numpy as np
from qtpy.QtCore import QObject, Signal
from BOB.checkpoints import checkpoints

class SegmentationWorker(QObject):
    """Worker class for running segmentation in a separate thread."""
    
    # Signals
    finished = Signal()
    error = Signal(str)
    progress = Signal(str)
    results_ready = Signal(object, object, bool)  # results, names/prompts_new, is_3d
    
    def __init__(self, img, prompts, checkpoint, config, is_3d, image_layer_metadata):
        super().__init__()
        self.img = img
        self.prompts = prompts
        self.checkpoint = checkpoint
        self.config = config
        self.is_3d = is_3d
        self.image_layer_metadata = image_layer_metadata
        
    def run(self):
        """Run the prediction in the worker thread."""
        try:
            self.progress.emit("Initializing predictor...")
            
            if self.is_3d:
                # Check if the checkpoint is MedSAM or SAM2
                if self.config == checkpoints.MedSAM_cfg:
                    # Use MedSAM2Predictor3D
                    self.progress.emit("Running Segmentation using MedSAM2Predictor3D...")
                    from BOB.predictors.medsam_3d import MedSAM2Predictor3D
                    predictor = MedSAM2Predictor3D(checkpoint=self.checkpoint)
                elif self.config is None:
                    # Use nnInteractivePredictor3D
                    self.progress.emit("Running Segmentation using nnInteractivePredictor3D...")
                    from BOB.predictors.nnInteractive_3d import nnInteractivePredictor3D
                    predictor = nnInteractivePredictor3D()
                else:
                    self.progress.emit("Running Segmentation using SAM2Predictor3D...")
                    from BOB.predictors.sam2_3d import SAM2Predictor3D
                    predictor = SAM2Predictor3D(
                        checkpoint=self.checkpoint,
                        config=self.config,
                    )

                results, names = predictor.predict(self.img, self.prompts)
                self.results_ready.emit(results, names, True)
                
            else:
                self.progress.emit("Running Segmentation using SAM2ImageSegmenter...")
                from BOB.predictors.sam2_2d import SAM2ImageSegmenter
                predictor = SAM2ImageSegmenter(
                    checkpoint=self.checkpoint,
                    config=self.config,
                )
                
                segmasks, scores, prompts_new = predictor.predict(self.img, self.prompts)
                self.results_ready.emit(segmasks, prompts_new, False)
                

            self.progress.emit("Segmentation completed!")
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()

class PromptGenerationWorker(QObject):
    """Worker class for running prompt generation in a separate thread."""
    
    # Signals
    finished = Signal()
    error = Signal(str)
    progress = Signal(str)
    prompts_ready = Signal(list)  # Generated prompts
    
    def __init__(self, generator, img, confidence, iou_threshold, max_detections, 
                 n_prompts_per_obj, multiprompt_z_spacing, max_z_distance, allowed_classes, allow_multiobject_3d):
        super().__init__()
        self.generator = generator
        self.img = img
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.n_prompts_per_obj = n_prompts_per_obj
        self.multiprompt_z_spacing = multiprompt_z_spacing
        self.max_z_distance = max_z_distance
        self.allowed_classes = allowed_classes
        self.allow_multiobject_3d = allow_multiobject_3d
        
    def run(self):
        """Run the prompt generation in the worker thread."""
        try:
            self.progress.emit("Generating prompts...")
            
            prompts = self.generator.generate_prompt(
                self.img,
                None,
                confidence=self.confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                n_prompts_per_obj=self.n_prompts_per_obj,
                multiprompt_z_spacing=self.multiprompt_z_spacing,
                max_z_distance=self.max_z_distance,
                plot_prompts=False,  # Disable plotting
                allowed_classes=self.allowed_classes,
                allow_multiobject_3d=self.allow_multiobject_3d
            )
            
            self.progress.emit(f"Generated {len(prompts)} prompts!")
            self.prompts_ready.emit(prompts)
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()