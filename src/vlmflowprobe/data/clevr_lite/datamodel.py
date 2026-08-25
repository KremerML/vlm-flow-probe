from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Object:
    """Single object in a scene"""
    id: int
    shape: str
    color: str
    size: str
    position: Tuple[float, float]  # (x, y) in [0, 1]
    pixel_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class Scene:
    """Complete scene specification"""
    objects: List[Object]
    image_path: str
    scene_id: int


@dataclass
class Question:
    """Question-answer pair with metadata"""
    scene_id: int
    question: str
    answer: str
    question_type: str
    image_path: str
    scene_objects: List[Dict]
    template_id: int
    is_held_out_combo: bool
