"""
Physical Metric Dimensions and Priors for Robotics Navigation Objects.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ObjectDimension:
    """Real-world 3D metric dimensions in meters."""
    height_m: float
    width_m: float
    depth_m: Optional[float] = None
    min_dist_m: float = 0.2
    max_dist_m: float = 40.0
    aspect_ratio: float = 1.0

    def __post_init__(self):
        if self.width_m > 0:
            self.aspect_ratio = self.height_m / self.width_m


class ObjectPriors:
    """Registry of known physical dimensions for navigation-relevant objects."""

    # Default dimensions based on standard highway/industrial specifications (MUTCD / OSHA / Robotics ISO)
    _REGISTRY: Dict[str, ObjectDimension] = {
        "cone": ObjectDimension(
            height_m=0.70,        # Standard 28-inch traffic cone (0.70m)
            width_m=0.38,         # Standard base width
            depth_m=0.38,
            min_dist_m=0.3,
            max_dist_m=25.0
        ),
        "traffic cone": ObjectDimension(
            height_m=0.70,
            width_m=0.38,
            depth_m=0.38,
            min_dist_m=0.3,
            max_dist_m=25.0
        ),
        "barrier": ObjectDimension(
            height_m=0.90,        # Standard construction barrier / Jersey barricade
            width_m=1.80,         # Standard unit width
            depth_m=0.60,
            min_dist_m=0.5,
            max_dist_m=35.0
        ),
        "traffic barrier": ObjectDimension(
            height_m=0.90,
            width_m=1.80,
            depth_m=0.60,
            min_dist_m=0.5,
            max_dist_m=35.0
        ),
        "stop sign": ObjectDimension(
            height_m=0.75,        # Standard 30-inch (0.75m) octagonal stop sign
            width_m=0.75,
            depth_m=0.05,
            min_dist_m=0.5,
            max_dist_m=50.0
        ),
        "stop_sign": ObjectDimension(
            height_m=0.75,
            width_m=0.75,
            depth_m=0.05,
            min_dist_m=0.5,
            max_dist_m=50.0
        ),
    }

    @classmethod
    def get(cls, class_name: str) -> Optional[ObjectDimension]:
        """Lookup dimension by normalized class name."""
        key = class_name.strip().lower()
        return cls._REGISTRY.get(key)

    @classmethod
    def register(cls, class_name: str, dimension: ObjectDimension) -> None:
        """Register or override dimensions for a given class."""
        cls._REGISTRY[class_name.strip().lower()] = dimension
