"""
A napari plugin for medical prompt generation and visualization using BOB (Bounding-box Oracle for Biomedicine)
and SAM (Segment Anything Model).
"""

__version__ = "1.0.0"

from .napari.napari_widget import create_bob_widget
from .napari.reader_extensions import napari_get_reader
from .prompt_generator import BOB

# Define the public API of the module
__all__ = [
    "create_bob_widget", # Required for napari plugin
    "napari_get_reader", # Optional for napari plugin
    "BOB", # The main prompt generator class
]