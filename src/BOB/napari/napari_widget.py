import numpy as np
from qtpy.QtCore import QThread, Signal, QObject
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QSpinBox,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QScrollArea,
    QComboBox,
    QFrame,
    QDoubleSpinBox,
)
import napari
from napari.layers import Image, Shapes
from napari.utils.notifications import show_info, show_warning, show_error

from BOB.checkpoints import checkpoints
from BOB.inference import utils
from BOB.prompt_generator import BOB
from BOB.napari.runner_threads import SegmentationWorker, PromptGenerationWorker
from BOB.napari.presets import presets


class SupportedClassesWidget(QWidget):
    def __init__(self, supported_classes, parent=None):
        super().__init__(parent)
        self.supported_classes = supported_classes
        self.checkboxes = {}
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)  # Limit height
        scroll_area.setMinimumHeight(100)   # Minimum height
        scroll_area.setFrameStyle(QFrame.Box)
        
        # Create widget to hold checkboxes
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add checkboxes for each supported class
        for class_id, class_name in self.supported_classes.items():
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)  # Default checked state
            # Store both the checkbox and the class_id for easy access
            checkbox.class_id = class_id
            self.checkboxes[class_id] = checkbox
            checkbox_layout.addWidget(checkbox)
        
        # Add stretch at the end to push checkboxes to top of scroll area
        checkbox_layout.addStretch()
        
        scroll_area.setWidget(checkbox_widget)
        main_layout.addWidget(scroll_area)
        
        # Create buttons layout
        button_layout = QHBoxLayout()
        
        # Select All button
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        # Deselect All button  
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)
        
        main_layout.addLayout(button_layout)
        
    def select_all(self):
        """Check all checkboxes"""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
            
    def deselect_all(self):
        """Uncheck all checkboxes"""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
        
    def get_selected_ids(self):
        """Return a list of selected class IDs"""
        return [class_id for class_id, checkbox in self.checkboxes.items() 
                if checkbox.isChecked()]

class PromptVisualizer(QWidget):
    """
    This widget allows users to visualize and generate prompts for medical images in napari.
    It also provides a connection to the MedSAM2Predictor3D or SAM2ImageSegmenter for direct
    image segmentation based on the generated prompts or manually added shapes.
    """
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.generator = BOB()
        self.supported_classes = self.generator.get_supported_classes() # Dictionary of class_id:class_label
        self.prompt_layers = []
        self.segmentation_layers = []
        self.image_layer = None  # To store the currently selected image layer
        self.original_image_layer = None  # To store the original (non-windowed) image layer reference
        self.original_image_data = None  # Store original image data for windowing
        self.windowed_image_layer = None  # Store windowed image layer reference
        self.is_3d = False

        # Threading components
        self.segmentation_thread = None
        self.segmentation_worker = None
        self.prompt_generation_thread = None
        self.prompt_generation_worker = None

        # Flag to prevent switching to Custom when applying presets
        self.applying_preset = False

        # Connect to layer removal events to clean up the lists tracking all prompt and segmentation layers
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        
        # Connect to layer events to automatically update image layer when layers change
        self.viewer.layers.events.inserted.connect(self._on_layer_added)
        self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_changed)

        total_layout = QVBoxLayout()

        # Preset dropdown menu
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_selector = QComboBox()
        self.preset_selector.addItem("Custom")
        for preset_name in presets.keys():
            self.preset_selector.addItem(preset_name)
        self.preset_selector.setToolTip(
            "Select a preset configuration for common medical imaging tasks. "
            "This will automatically adjust all parameters for optimal results."
        )
        self.preset_selector.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_selector)
        total_layout.addLayout(preset_layout)

        # Prompt Generation Model selector
        prompt_model_layout = QHBoxLayout()
        prompt_model_layout.addWidget(QLabel("Prompt Generation Model:"))
        self.prompt_model_selector = QComboBox()
        self.prompt_model_selector.addItem("D-FINE-Nano")
        self.prompt_model_selector.addItem("YOLOv12-Nano")
        self.prompt_model_selector.addItem("YOLOv12-Small")
        self.prompt_model_selector.setToolTip(
            "Select the model to use for prompt generation. "
        )
        self.prompt_model_selector.currentTextChanged.connect(self.on_prompt_model_changed)
        self.prompt_model_selector.currentTextChanged.connect(self.on_parameter_changed)
        prompt_model_layout.addWidget(self.prompt_model_selector)
        total_layout.addLayout(prompt_model_layout)

        self.on_prompt_model_changed()

        # Add spacer
        total_layout.addSpacing(10)

        # Windowing controls
        windowing_layout = QVBoxLayout()
        
        # Windowing preset dropdown
        windowing_preset_layout = QHBoxLayout()
        windowing_preset_layout.addWidget(QLabel("Windowing Preset:"))
        self.windowing_preset_selector = QComboBox()
        self.windowing_preset_selector.addItem("No Windowing", {"wl": 0, "ww": 0})
        self.windowing_preset_selector.addItem("Abdomen", {"wl": 60, "ww": 400})
        self.windowing_preset_selector.addItem("Bone", {"wl": 300, "ww": 1500})
        self.windowing_preset_selector.addItem("Brain", {"wl": 40, "ww": 80})
        self.windowing_preset_selector.addItem("Chest", {"wl": -600, "ww": 1600})
        self.windowing_preset_selector.addItem("Liver", {"wl": 60, "ww": 160})
        self.windowing_preset_selector.addItem("Lung", {"wl": -600, "ww": 1500})
        self.windowing_preset_selector.addItem("Mediastinum", {"wl": 50, "ww": 350})
        self.windowing_preset_selector.addItem("Spine", {"wl": 50, "ww": 250})
        self.windowing_preset_selector.addItem("Custom", {"wl": 0, "ww": 0})
        self.windowing_preset_selector.setToolTip(
            "Select windowing preset for CT images. Different presets optimize contrast for different tissue types."
        )
        self.windowing_preset_selector.currentTextChanged.connect(self.apply_windowing_preset)
        windowing_preset_layout.addWidget(self.windowing_preset_selector)
        windowing_layout.addLayout(windowing_preset_layout)

        # Window Level and Window Width controls
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("Window Level (WL):"))
        self.window_level = QSpinBox()
        self.window_level.setRange(-1024, 3071)  # Typical CT range
        self.window_level.setValue(0)
        self.window_level.setToolTip("Window Level (center) for CT windowing")
        self.window_level.valueChanged.connect(self.on_windowing_parameter_changed)
        wl_layout.addWidget(self.window_level)
        windowing_layout.addLayout(wl_layout)

        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel("Window Width (WW):"))
        self.window_width = QSpinBox()
        self.window_width.setRange(1, 4095)  # Window width must be positive
        self.window_width.setValue(1)
        self.window_width.setToolTip("Window Width for CT windowing")
        self.window_width.valueChanged.connect(self.on_windowing_parameter_changed)
        ww_layout.addWidget(self.window_width)
        windowing_layout.addLayout(ww_layout)

        total_layout.addLayout(windowing_layout)

        # Add spacer
        total_layout.addSpacing(10)

        # Parameter controls for prompt generation
        for name, display_name, mn, mx, default, control_type, tooltip in [
            ("confidence", "Confidence Threshold", 0.1, 1.0, 0.5, "double", 
             "Confidence threshold for YOLO detection. Higher values mean fewer but higher quality detections."),
            ("iou_threshold", "IoU Threshold", 0.1, 1.0, 0.7, "double", 
             "IoU threshold for non-max suppression. Lower values mean less overlap allowed between boxes."),
            ("max_detections", "Max Detections", 1, 100, 20, "int", 
             "Maximum number of detections per 2D slice."),
            ("n_prompts_per_obj", "Prompts per Object", 1, 20, 1, "int", 
             "Number of prompts to generate per detected object (in 3D)."),
            ("multiprompt_z_spacing", "Multi-prompt Z Spacing", 1, 50, 10, "int", 
             "Minimum Z distance between multiple prompts for the same object (in 3D)."),
            ("max_z_distance", "Max Z Distance", 1, 100, 10, "int", 
             "Maximum Z distance for clustering prompts into the same object (in 3D)."),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{display_name}:"))
            
            if control_type == "double":
                control = QDoubleSpinBox()
                control.setRange(mn, mx)
                control.setValue(default)
                control.setSingleStep(0.1)
                control.setDecimals(2)
            else:  # int
                control = QSpinBox()
                control.setRange(mn, mx)
                control.setValue(default)
            
            control.setToolTip(tooltip)
            row.addWidget(control)
            setattr(self, name, control)
            
            # Connect value changed signals to switch to Custom when user modifies
            if control_type == "double":
                control.valueChanged.connect(self.on_parameter_changed)
            else:
                control.valueChanged.connect(self.on_parameter_changed)
            
            total_layout.addLayout(row)

        # Allow Multiobject 3D checkbox
        self.allow_multiobject_3d = QCheckBox("Allow Multiobject 3D")
        self.allow_multiobject_3d.setChecked(True)  # Default to True
        self.allow_multiobject_3d.setToolTip(
            "If disabled, all objects of the same class receive the same object ID, which is useful for "
            "organ segmentation where each organ typically exists only once (e.g., liver, heart, etc.)."
        )
        self.allow_multiobject_3d.stateChanged.connect(self.on_parameter_changed)
        total_layout.addWidget(self.allow_multiobject_3d)

        # Enable custom class prediction checkbox
        self.predict_all_classes = QCheckBox("Predict all classes")
        self.predict_all_classes.setChecked(True)  # Default to True
        self.predict_all_classes.setToolTip("If unchecked, you can select which classes to predict.")
        self.predict_all_classes.stateChanged.connect(lambda state: self.supported_classes_widget.setVisible(state == 0))
        self.predict_all_classes.stateChanged.connect(self.on_parameter_changed)
        total_layout.addWidget(self.predict_all_classes)

        # Supported classes widget
        self.supported_classes_widget = SupportedClassesWidget(
            self.supported_classes
        )
        self.supported_classes_widget.setMinimumHeight(100)  # Set minimum height
        self.supported_classes_widget.setMaximumHeight(300)  # Set maximum height
        self.supported_classes_widget.setVisible(False)  # Initially visible
        total_layout.addWidget(self.supported_classes_widget)

        # Button to add a new prompt layer
        self.add_btn = QPushButton("Add Prompt manually")
        self.add_btn.clicked.connect(self.create_empty_shapes_layer)
        total_layout.addWidget(self.add_btn)

        # TEMPORARY DEBUG BUTTON
        # self.temp_button = QPushButton("PRINT PROMPT LAYER SHAPES")
        # self.temp_button.clicked.connect(self.on_print_prompt_layer_shapes)
        # total_layout.addWidget(self.temp_button)

        # Horizontal layout to generate/clear prompts
        btn_layout = QHBoxLayout()
        self.gen_btn = QPushButton("Generate Prompts")
        self.gen_btn.setMinimumSize(120, 35)  # Make button more rectangular
        clr_btn = QPushButton("Clear Prompts")
        clr_btn.setMinimumSize(120, 35)  # Make button more rectangular
        self.gen_btn.clicked.connect(self.on_generate)
        clr_btn.clicked.connect(self.on_clear_prompts)
        btn_layout.addWidget(self.gen_btn)
        btn_layout.addWidget(clr_btn)
        total_layout.addLayout(btn_layout)

        # Add spacer between prompt buttons and model selector
        total_layout.addSpacing(15)

        # Model selector with label
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Segmentation Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItem("MedSAM2_latest", (checkpoints.MedSAM2_latest, checkpoints.MedSAM_cfg))
        self.model_selector.addItem("MedSAM2_CTLesion", (checkpoints.MedSAM2CT, checkpoints.MedSAM_cfg))
        self.model_selector.addItem("MedSAM2_MRI_LiverLesion", (checkpoints.MedSAM2_Liver, checkpoints.MedSAM_cfg))
        self.model_selector.addItem("MedSAM2_US_Heart", (checkpoints.MedSAM2_Heart, checkpoints.MedSAM_cfg))
        self.model_selector.addItem("MedSAM2_2411", (checkpoints.MedSAM2_2411, checkpoints.MedSAM_cfg))
        self.model_selector.addItem("SAM2.1 Tiny", (checkpoints.SAM2_tiny, checkpoints.SAM2_tiny_cfg))
        self.model_selector.addItem("SAM2.1 Small", (checkpoints.SAM2_small, checkpoints.SAM2_small_cfg))
        self.model_selector.addItem("SAM2.1 Base+", (checkpoints.SAM2_base, checkpoints.SAM2_base_cfg))
        self.model_selector.addItem("SAM2.1 Large", (checkpoints.SAM2_large, checkpoints.SAM2_large_cfg))
        self.model_selector.setToolTip(
            "Select the model to use for segmentation. "
            "MedSAM2 models are optimized for medical images."
        )
        self.model_selector.currentTextChanged.connect(self.on_parameter_changed)
        model_layout.addWidget(self.model_selector)
        total_layout.addLayout(model_layout)

        # Horizontal layout for segmentation buttons
        segment_btn_layout = QHBoxLayout()
        self.segment_btn = QPushButton("Segment Prompts")
        self.segment_btn.setMinimumSize(120, 35)  # Make button more rectangular
        self.segment_btn.clicked.connect(self.on_segment)
        self.segment_btn.setEnabled(False)  # Initially disabled
        
        self.clear_segmentation_btn = QPushButton("Clear Segmentation")
        self.clear_segmentation_btn.setMinimumSize(120, 35)  # Make button more rectangular
        self.clear_segmentation_btn.clicked.connect(self.on_clear_segment)
        self.clear_segmentation_btn.setEnabled(False)
        
        segment_btn_layout.addWidget(self.segment_btn)
        segment_btn_layout.addWidget(self.clear_segmentation_btn)
        total_layout.addLayout(segment_btn_layout)

        # Set the widget's layout at the very end
        self.setLayout(total_layout)

    def apply_preset(self, preset_name):
        """Apply preset configurations based on the selected preset."""
        if preset_name == "Custom":
            return  # Don't change anything for custom
        
        # Set flag to prevent switching back to Custom during preset application
        self.applying_preset = True
        
        
        
        if preset_name in presets:
            preset = presets[preset_name]
            
            # Apply parameter values
            self.confidence.setValue(preset["confidence"])
            self.iou_threshold.setValue(preset["iou_threshold"])
            self.max_detections.setValue(preset["max_detections"])
            self.n_prompts_per_obj.setValue(preset["n_prompts_per_obj"])
            self.multiprompt_z_spacing.setValue(preset["multiprompt_z_spacing"])
            self.max_z_distance.setValue(preset["max_z_distance"])
            self.allow_multiobject_3d.setChecked(preset["allow_multiobject_3d"])
            self.predict_all_classes.setChecked(preset["predict_all_classes"])
            
            # Set model selection
            model_name = preset["model"]
            for i in range(self.model_selector.count()):
                if model_name in self.model_selector.itemText(i):
                    self.model_selector.setCurrentIndex(i)
                    break
            
            # Set prompt generation model selection
            prompt_model_name = preset["prompt_model"]
            for i in range(self.prompt_model_selector.count()):
                if prompt_model_name in self.prompt_model_selector.itemText(i):
                    self.prompt_model_selector.setCurrentIndex(i)
                    break
            
            # Set windowing preset if specified
            if "windowing_preset" in preset:
                windowing_preset_name = preset["windowing_preset"]
                for i in range(self.windowing_preset_selector.count()):
                    if windowing_preset_name == self.windowing_preset_selector.itemText(i):
                        self.windowing_preset_selector.setCurrentIndex(i)
                        break
            
            show_info(f"Applied preset: {preset_name}")
        
        # Reset flag to allow normal parameter change detection
        self.applying_preset = False

    def on_prompt_model_changed(self):
        """Handle prompt generation model change."""
        selected_model = self.prompt_model_selector.currentText()
        try:
            self.generator.switch_model(selected_model)
            show_info(f"Switched to {selected_model} model for prompt generation")
        except Exception as e:
            show_error(f"Failed to switch model: {e}")

    def on_parameter_changed(self):
        """Switch preset to Custom when user manually changes any parameter."""
        # Don't switch to Custom if we're currently applying a preset
        if not self.applying_preset:
            # Temporarily disconnect the signal to avoid infinite loop
            self.preset_selector.currentTextChanged.disconnect()
            self.preset_selector.setCurrentText("Custom")
            self.preset_selector.currentTextChanged.connect(self.apply_preset)

    def get_selected_classes(self):
        """
        Returns a dictionary of selected class IDs and their names.
        If 'Predict all classes' is checked, returns None to indicate all classes.
        """
        if self.predict_all_classes.isChecked():
            return None
        else:
            return self.supported_classes_widget.get_selected_ids()
        

    def create_empty_shapes_layer(self):
        """Create an empty shapes layer for a single shape and track it."""

        # Initialize prompt_layers if it doesn't exist
        if not hasattr(self, "prompt_layers"):
            self.prompt_layers = []

        self.set_image_layer()  # Ensure we have the current image layer

        # Determine ndim based on current image
        ndim = 3 if self.is_3d else 2

        # Generate unique layer name
        layer_count = len(self.prompt_layers)
        layer_name = f"Shape_{layer_count + 1}"

        # Create empty shapes layer
        shapes_layer = self.viewer.add_shapes(
            name=layer_name,
            ndim=ndim,
            edge_color="red",
            face_color="transparent",
            edge_width=3,
            scale=(self.original_image_layer or self.image_layer).metadata.get("spacing", None),
            rotate=(self.original_image_layer or self.image_layer).metadata.get("direction", None),
            translate=(self.original_image_layer or self.image_layer).metadata.get("origin", None),
            metadata={
                "origin": (self.original_image_layer or self.image_layer).metadata.get("origin", None),
                "direction": (self.original_image_layer or self.image_layer).metadata.get("direction", None),
                "spacing": (self.original_image_layer or self.image_layer).metadata.get("spacing", None),
            },
        )

        # Add the layer directly to prompt_layers
        self.prompt_layers.append(shapes_layer)

        # Optionally connect to events if you want to know when shapes are added to this layer
        shapes_layer.events.data.connect(
            lambda event: self._on_shape_added_to_layer(shapes_layer, event)
        )

        shapes_layer.mode = "add_rectangle"  # Set the mode to rectangle for new shapes

        return shapes_layer

    def _on_shape_added_to_layer(self, layer, event):
        """Optional callback when a shape is added to a specific layer."""
        if len(layer.data) > 0:
            show_info(f"Shape added to layer: {layer.name}")
            self.segment_btn.setEnabled(
                True
            )  # Enable segmentation button after generating prompts

            # Change mode to the move tool if a shape is added
            layer.mode = "pan_zoom"  # Set the mode to pan_zoom for existing shapes

        # Enable rectangle button if the shape was deleted
        elif len(layer.data) == 0:
            show_info(
                f"Shape removed from layer: {layer.name}. Rectangle button enabled again."
            )
            layer.mode = "add_rectangle"  # Reset mode to rectangle for new shapes

    def _on_layer_removed(self, event):
        """
        Handle when a layer is manually removed from napari.
        Remove the layer from our tracking lists if it exists.
        """
        removed_layer = event.value
        
        # Check if the removed layer is in prompt_layers
        if removed_layer in self.prompt_layers:
            self.prompt_layers.remove(removed_layer)
            # If no prompt layers remain, disable the segment button
            if not self.prompt_layers:
                self.segment_btn.setEnabled(False)
        
        # Check if the removed layer is in segmentation_layers
        if removed_layer in self.segmentation_layers:
            self.segmentation_layers.remove(removed_layer)
            # If no segmentation layers remain, disable the clear segmentation button
            if not self.segmentation_layers:
                self.clear_segmentation_btn.setEnabled(False)
        
        # Check if the removed layer is the windowed image layer
        if removed_layer == self.windowed_image_layer:
            self.windowed_image_layer = None
        
        # Check if the removed layer is the main image layer
        if removed_layer == self.image_layer:
            self.image_layer = None
            # Only clear original references if the removed layer was the original
            if removed_layer == self.original_image_layer:
                self.original_image_data = None
                self.original_image_layer = None
        
        # Check if the removed layer is the original image layer
        if removed_layer == self.original_image_layer:
            self.original_image_layer = None
            self.original_image_data = None

    def _on_layer_added(self, event):
        """
        Handle when a new layer is added to napari.
        If it's an image layer, automatically set it as the current image layer.
        """
        added_layer = event.value
        if isinstance(added_layer, Image) and "_windowed" not in added_layer.name:
            # Only auto-select non-windowed image layers
            # Automatically set the newly added image layer as the current one
            self.viewer.layers.selection.active = added_layer
            self.set_image_layer()
            show_info(f"Automatically selected new image layer: {added_layer.name}")

    def _on_layer_selection_changed(self, event):
        """
        Handle when the layer selection changes in napari.
        If an image layer is selected, automatically set it as the current image layer.
        """
        if self.viewer.layers.selection.active and isinstance(self.viewer.layers.selection.active, Image):
            self.set_image_layer()
            layer_name = self.viewer.layers.selection.active.name
            if "_windowed" in layer_name:
                show_info(f"Selected windowed image layer: {layer_name}")
            else:
                show_info(f"Selected image layer: {layer_name}")

    def apply_windowing_preset(self, preset_name):
        """Apply windowing preset configurations."""
        if preset_name == "Custom":
            return  # Don't change anything for custom
        
        preset_data = self.windowing_preset_selector.currentData()
        if preset_data:
            # Temporarily disconnect signals to avoid triggering custom
            self.window_level.valueChanged.disconnect()
            self.window_width.valueChanged.disconnect()
            
            self.window_level.setValue(preset_data["wl"])
            self.window_width.setValue(preset_data["ww"])
            
            # Reconnect signals
            self.window_level.valueChanged.connect(self.on_windowing_parameter_changed)
            self.window_width.valueChanged.connect(self.on_windowing_parameter_changed)
            
            # Apply windowing automatically for presets (except "No Windowing")
            if preset_name != "No Windowing":
                self.apply_windowing()
            else:
                self.reset_windowing()

    def on_windowing_parameter_changed(self):
        """Switch windowing preset to Custom when user manually changes parameters and apply windowing automatically."""
        # Temporarily disconnect the signal to avoid infinite loop
        self.windowing_preset_selector.currentTextChanged.disconnect()
        self.windowing_preset_selector.setCurrentText("Custom")
        self.windowing_preset_selector.currentTextChanged.connect(self.apply_windowing_preset)
        
        # Automatically apply windowing when parameters change
        self.apply_windowing()

    def apply_windowing(self):
        """Apply windowing to the current image."""
        if self.original_image_data is None or self.original_image_layer is None:
            show_warning("No original image available for windowing.")
            return

        wl = self.window_level.value()
        ww = self.window_width.value()
        
        if ww == 0:
            show_warning("Window Width cannot be zero.")
            return

        # Always apply windowing to the original image data, not to windowed data
        windowed_data = self.window_image(self.original_image_data, wl, ww)
        
        # Update or create windowed image layer
        self.update_windowed_image_layer(windowed_data)
        
        show_info(f"Applied windowing: WL={wl}, WW={ww}")

    def reset_windowing(self):
        """Reset to original image without windowing."""
        if self.original_image_data is None or self.original_image_layer is None:
            show_warning("No original image data available.")
            return
            
        # Remove windowed layer if it exists
        if self.windowed_image_layer and self.windowed_image_layer in self.viewer.layers:
            self.viewer.layers.remove(self.windowed_image_layer)
            self.windowed_image_layer = None
        
        # Simply make the original layer visible again without modifying its data or transforms
        # The original layer should already have the correct positioning
        self.original_image_layer.visible = True
        self.img = self.original_image_data
        
        # Update current image layer reference
        self.image_layer = self.original_image_layer
        
        # Select the original layer
        self.viewer.layers.selection.active = self.original_image_layer
        
        show_info("Reset to original image (no windowing)")

    def window_image(self, image, window_level, window_width):
        """
        Apply windowing to an image.
        
        Args:
            image: Input image array
            window_level: Center of the window
            window_width: Width of the window
            
        Returns:
            Windowed image normalized to 0-255 range
        """
        # Calculate window boundaries
        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2
        
        # Apply windowing
        windowed = np.clip(image, window_min, window_max)
        
        # Normalize to 0-255 range
        if window_max != window_min:
            windowed = ((windowed - window_min) / (window_max - window_min)) * 255
        else:
            windowed = np.zeros_like(windowed)
            
        return windowed.astype(np.uint8)

    def update_windowed_image_layer(self, windowed_data):
        """Update or create a windowed image layer in napari."""
        if self.original_image_layer is None:
            return
            
        # Copy metadata from original layer (not potentially windowed current layer)
        layer_kwargs = {
            'name': f"{self.original_image_layer.name}_windowed",
            'metadata': self.original_image_layer.metadata.copy() if hasattr(self.original_image_layer, 'metadata') else {},
            'scale': self.original_image_layer.metadata.get("spacing", None) if hasattr(self.original_image_layer, 'metadata') else self.original_image_layer.scale,
            'translate': self.original_image_layer.metadata.get("origin", None) if hasattr(self.original_image_layer, 'metadata') else self.original_image_layer.translate,
            'rotate': self.original_image_layer.metadata.get("direction", None) if hasattr(self.original_image_layer, 'metadata') else self.original_image_layer.rotate,
            'opacity': self.original_image_layer.opacity,
            'blending': self.original_image_layer.blending,
            'visible': True  # Always make the windowed layer visible
        }
        
        # Remove the original windowed layer if it exists
        if self.windowed_image_layer and self.windowed_image_layer in self.viewer.layers:
            self.viewer.layers.remove(self.windowed_image_layer)
        
        # Add new windowed layer
        self.windowed_image_layer = self.viewer.add_image(windowed_data, **layer_kwargs)
        
        # Update working image data
        self.img = windowed_data
        
        # Hide original layer and show windowed layer (don't modify original layer transforms)
        self.original_image_layer.visible = False
        
        # Select the windowed layer
        self.viewer.layers.selection.active = self.windowed_image_layer

            
    def set_image_layer(self):
        # Get the currently selected layer
        selected_layers = list(self.viewer.layers.selection)
        if not selected_layers:
            show_warning("No layer selected. Please select an image layer.")
            return

        layer = selected_layers[-1]  # Use the most recently selected layer
        if not isinstance(layer, Image):
            show_warning(
                "Selected layer is not an image. Please select an image layer."
            )
            return
        
        # Check if this is a windowed layer
        if "_windowed" in layer.name:
            # If it's a windowed layer, don't update the original references
            # Just update the current working layer
            self.image_layer = layer
            self.img = layer.data
        else:
            # This is an original (non-windowed) layer
            # Store original image data and layer reference for windowing
            self.original_image_data = layer.data.copy()
            self.original_image_layer = layer
            self.image_layer = layer
            
            # Get the image data from the layer
            img = layer.data

            # If the data is not in the range of 0-255, normalize it
            if min(np.unique(img)) < 0 or max(np.unique(img)) > 255:
                img = utils.normalize_minmax(img)

            self.img = img

        self.is_3d = (self.img.ndim == 3 and self.img.shape[-1] != 3)  # Check if it's a 3D image

    def on_generate(self):
        # Don't start a new generation if one is already running
        if self.prompt_generation_thread and self.prompt_generation_thread.isRunning():
            show_warning("Prompt generation is already running. Please wait for it to complete.")
            return

        self.on_clear_prompts()
        self.set_image_layer()  # Ensure we have the current image layer

        if not hasattr(self, 'img') or self.img is None:
            show_error("No image selected. Please select an image layer.")
            return

        # Disable the generate button during generation
        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("Generating...")
        
        # Create worker and thread
        self.prompt_generation_worker = PromptGenerationWorker(
            generator=self.generator,
            img=self.img,
            confidence=self.confidence.value(),
            iou_threshold=self.iou_threshold.value(),
            max_detections=self.max_detections.value(),
            n_prompts_per_obj=self.n_prompts_per_obj.value(),
            multiprompt_z_spacing=self.multiprompt_z_spacing.value(),
            max_z_distance=self.max_z_distance.value(),
            allowed_classes=self.get_selected_classes(),
            allow_multiobject_3d=self.allow_multiobject_3d.isChecked()
        )
        
        self.prompt_generation_thread = QThread()
        self.prompt_generation_worker.moveToThread(self.prompt_generation_thread)
        
        # Connect signals
        self.prompt_generation_thread.started.connect(self.prompt_generation_worker.run)
        self.prompt_generation_worker.finished.connect(self.prompt_generation_thread.quit)
        self.prompt_generation_worker.finished.connect(self._on_generation_finished)
        self.prompt_generation_worker.error.connect(lambda msg: show_error(f"Prompt generation failed: {msg}"))
        self.prompt_generation_worker.progress.connect(lambda msg: show_info(msg))
        self.prompt_generation_worker.prompts_ready.connect(self._on_prompts_ready)
        
        # Start the thread
        show_info("Starting prompt generation...")
        self.prompt_generation_thread.start()

    def _on_generation_finished(self):
        """Handle prompt generation completion."""
        # Re-enable the generate button
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate Prompts")
        
        # Clean up thread and worker
        if self.prompt_generation_thread:
            self.prompt_generation_thread.deleteLater()
            self.prompt_generation_thread = None
        if self.prompt_generation_worker:
            self.prompt_generation_worker.deleteLater()
            self.prompt_generation_worker = None

    def get_transform_metadata(self, image_layer, ndim):
        """Extract transform metadata from a napari layer."""
        # Prefer original layer metadata if available
        layer_to_use = self.original_image_layer or image_layer
        metadata = layer_to_use.metadata if hasattr(layer_to_use, 'metadata') else {}

        origin = metadata.get("origin", None)
        direction = metadata.get("direction", None)
        spacing = metadata.get("spacing", None)

        if origin is None:
            return layer_to_use.translate, layer_to_use.rotate, layer_to_use.scale
        elif len(origin) != ndim:
            return layer_to_use.translate, layer_to_use.rotate, layer_to_use.scale
        else:
            return origin, direction, spacing

    def _on_prompts_ready(self, prompts):
        """Handle generated prompts and create napari layers."""
        try:
            if not prompts:
                show_warning("No prompts generated.")
                return

            # Create unique colors for each object ID
            ids = list({p.obj_id for p in prompts})
            colors = {i: np.random.rand(3) for i in ids}

            for p in prompts:
                try:
                    box = p.box
                    if self.is_3d:
                        coords = [
                            (p.z, box[1], box[0]),
                            (p.z, box[1], box[2]),
                            (p.z, box[3], box[2]),
                            (p.z, box[3], box[0]),
                        ]
                        ndim = 3
                    else:
                        coords = [
                            (box[1], box[0]),
                            (box[1], box[2]),
                            (box[3], box[2]),
                            (box[3], box[0]),
                        ]
                        ndim = 2

                    # Apply scaling and direction from metadata if available
                    coords_arr = np.array(coords, dtype=np.float32)

                    origin, direction, spacing = self.get_transform_metadata(self.image_layer, ndim)

                    shapes = Shapes(
                        data=[coords_arr.tolist()],
                        name=f"Obj {p.obj_id}: {p.class_label}",
                        ndim=ndim,
                        text=[f"{p.class_label} {p.obj_id} ({p.confidence:.2f})"],
                        edge_color=colors[p.obj_id],
                        face_color="transparent",
                        edge_width=4,
                        rotate=direction,
                        translate=origin,
                        scale=spacing,
                    )
                    shapes.metadata = {
                        "origin": origin,
                        "direction": direction,
                        "spacing": spacing,
                    }
                    
                    shapes.prompt = (
                        p  # Store the prompt in the shape for later reference
                    )
                    self.viewer.add_layer(shapes)
                    self.prompt_layers.append(shapes)

                except Exception as e:
                    show_error(f"Error creating shape for prompt {p.obj_id}: {e}")
                    # Full error traceback can be printed to console for debugging
                    import traceback
                    traceback.print_exc()
                    continue

            show_info(f"Successfully created {len(self.prompt_layers)} prompt layers.")
            self.segment_btn.setEnabled(
                True
            )  # Enable segmentation button after generating prompts
            
        except Exception as e:
            show_error(f"Error processing generated prompts: {e}")

    def convert_promptlayer_to_prompts(self) -> list[utils.Prompt]:
        prompts = []

        if not self.prompt_layers:
            show_warning("No prompts to segment. Please generate prompts first.")
            return prompts
        
        for i, layer in enumerate(self.prompt_layers):
            # Layers generated by BOB already have prompts stored, so we can use them directly
            if hasattr(layer, 'prompt'):
                prompts.append(layer.prompt)
            # Shapes created by the user need to be converted to prompts
            elif isinstance(layer, Shapes):
                for shape in layer.data:
                    box, z = self.shape_to_bounding_box(shape)
                    prompts.append(
                        utils.Prompt(
                            box=np.array(box),
                            z=z,
                            class_label=layer.name,
                            obj_id=len(self.prompt_layers) + i + 1,  # Unique ID for each prompt
                        )
                    )
        return prompts
    
    def on_segment(self):
        # Don't start a new prediction if one is already running
        if self.segmentation_thread and self.segmentation_thread.isRunning():
            show_warning("Prediction is already running. Please wait for it to complete.")
            return

        # Convert all layers with shapes to prompt objects
        prompts = self.convert_promptlayer_to_prompts()

        # Get the selected model/config from the combobox
        checkpoint, config = self.model_selector.currentData()
        
        # Disable the segment button during prediction
        self.segment_btn.setEnabled(False)
        self.segment_btn.setText("Segmenting...")
        
        # Create worker and thread
        self.segmentation_worker = SegmentationWorker(
            img=self.img,
            prompts=prompts,
            checkpoint=checkpoint,
            config=config,
            is_3d=self.is_3d,
            image_layer_metadata=self.image_layer.metadata
        )
        
        self.segmentation_thread = QThread()
        self.segmentation_worker.moveToThread(self.segmentation_thread)
        
        # Connect signals
        self.segmentation_thread.started.connect(self.segmentation_worker.run)
        self.segmentation_worker.finished.connect(self.segmentation_thread.quit)
        self.segmentation_worker.finished.connect(self._on_prediction_finished)
        self.segmentation_worker.error.connect(lambda msg: show_error(f"Prediction failed: {msg}"))
        self.segmentation_worker.progress.connect(lambda msg: show_info(msg))
        self.segmentation_worker.results_ready.connect(self._on_prediction_results)
        
        # Start the thread
        show_info(f"Starting segmentation of {len(prompts)} prompts using {checkpoint.split('/')[-1]} model...")
        self.segmentation_thread.start()

    def _on_prediction_finished(self):
        """Handle prediction completion."""
        # Re-enable the segment button
        self.segment_btn.setEnabled(True)
        self.segment_btn.setText("Segment Prompts")
        
        # Clean up thread and worker
        if self.segmentation_thread:
            self.segmentation_thread.deleteLater()
            self.segmentation_thread = None
        if self.segmentation_worker:
            self.segmentation_worker.deleteLater()
            self.segmentation_worker = None

    def _on_prediction_results(self, results, names_or_prompts, is_3d):
        """Handle prediction results and create napari layers."""
        try:
            origin, direction, spacing = self.get_transform_metadata(self.image_layer, 3 if is_3d else 2)

            if is_3d:
                # Handle 3D results
                results_dict, names = results, names_or_prompts
                for obj_id, res in results_dict.items():
                    label = self.viewer.add_labels(
                        res,
                        name=names[obj_id],
                        metadata={
                            "spacing": spacing,
                            "direction": direction,
                            "origin": origin,
                        },
                        scale=spacing,
                        rotate=direction,
                        translate=origin,
                    )
                    self.segmentation_layers.append(label)
            else:
                # Handle 2D results
                segmasks, prompts_new = results, names_or_prompts
                for i, mask in enumerate(segmasks):
                    if mask is not None:
                        # Give the mask a random color based on the prompt's object ID
                        color = int(np.random.rand(1) * 255)
                        # Set all mask pixels to the color
                        mask = np.where(mask > 0, color, 0)
                        
                        label = self.viewer.add_labels(
                            mask,
                            name=f"{prompts_new[i].class_label} {prompts_new[i].obj_id}",
                        )
                        self.segmentation_layers.append(label)
            
            self.clear_segmentation_btn.setEnabled(True)
            show_info("Segmentation completed successfully!")
            
        except Exception as e:
            show_error(f"Error processing prediction results: {e}")

    def on_clear_prompts(self):
        """Deletes all prompt layers."""
        # Take a copy to avoid modifying the list while iterating
        for lyr in self.prompt_layers.copy(): 
            if lyr in self.viewer.layers:
                # It is removed from prompt_layers automatically by the callback function on_layer_removed
                self.viewer.layers.remove(lyr)
        # Disable segmentation button after deleting prompts
        self.segment_btn.setEnabled(False)  

    def on_clear_segment(self):
        """Deletes all segmentation layers."""
        # Take a copy to avoid modifying the list while iterating
        for layer in self.segmentation_layers.copy():
            if layer in self.viewer.layers:
                # It is removed from segmentation_layers automatically by the callback function on_layer_removed
                self.viewer.layers.remove(layer)
        self.clear_segmentation_btn.setEnabled(False)
        show_info("Cleared all segmentation layers.")

    def on_print_prompt_layer_shapes(self):
        """Print the shapes of the prompt layers to the console."""
        if not self.prompt_layers:
            show_warning("No prompt layers available.")
            return

        for layer in self.prompt_layers:
            original_prompt = layer.prompt if hasattr(layer, 'prompt') else None

            if isinstance(layer, Shapes):
                print(f"Layer: {layer.name}, Shapes: {len(layer.data)}")
                for shape in layer.data:
                    box = self.shape_to_bounding_box(shape)
                    print(f"Converted Box: {box}")

                    if original_prompt:
                        print(f"  Original Box: {original_prompt.box}")
            else:
                print(f"Layer: {layer.name} is not a Shapes layer.")

    def shape_to_bounding_box(self, arr):
        """
        Extract [[x1, y1, x2, y2]], z from input array.

        Input formats:
        - [[z, y1, x1], [z, y2, x1], [z, y2, x2], [z, y1, x2]] -> returns [x1, y1, x2, y2], z
        - [[y1, x1], [y2, x1], [y2, x2], [y1, x2]] -> returns [x1, y1, x2, y2], 0

        Args:
            arr: List of coordinate points

        Returns:
            tuple: ([[x1, y1, x2, y2]], z)
        """
        if len(arr) != 4:
            raise ValueError("Input array must contain exactly 4 points.")

        # Check if it's 3D format: [[z, y, x], ...]
        if len(arr[0]) == 3:
            # Extract z value (should be same for all points)
            z = arr[0][0]

            # Extract all x and y coordinates
            x_coords = [int(point[2]) for point in arr]  # x is at index 2
            y_coords = [int(point[1]) for point in arr]  # y is at index 1

            # Get bounding box
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            return ([x1, y1, x2, y2], int(z))

        # Check if it's 2D format: [[x, y], ...]
        elif len(arr[0]) == 2:
            # Extract all x and y coordinates
            x_coords = [int(point[1]) for point in arr]
            y_coords = [int(point[0]) for point in arr]

            # Get bounding box
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            return ([x1, y1, x2, y2], int(0))

        else:
            raise ValueError("Input array must contain exactly 4 points.")


def create_bob_widget():
    """Create the BOB widget."""
    import napari

    viewer = napari.current_viewer()
    return PromptVisualizer(viewer)
