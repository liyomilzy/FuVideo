"""Model package: FuVideo pipeline + wrapper exposing a simple text2video interface."""

from .model import Model, ModelType
from .text_to_video_pipeline import FuVideoPipeline

__all__ = ["Model", "ModelType", "FuVideoPipeline"]
