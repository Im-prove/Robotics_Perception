"""
Visualization Utilities for Robotic Object Detection and Distance Annotation.
Ensures annotations strictly conform to assignment specifications: 'Cone, 1.5m', 'Stop Sign, 3.2m'.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from ..detector.base_detector import Detection


class Visualizer:
    """Renders annotated bounding boxes, distance badges, and navigation HUD."""

    # High-contrast color palette (BGR)
    COLOR_MAP = {
        "cone": (0, 165, 255),       # Vibrant Orange
        "barrier": (0, 0, 255),       # Solid Red
        "stop sign": (30, 30, 220),   # Deep Crimson
        "stop_sign": (30, 30, 220),
    }
    DEFAULT_COLOR = (0, 255, 128)

    @classmethod
    def get_color(cls, class_name: str) -> Tuple[int, int, int]:
        return cls.COLOR_MAP.get(class_name.strip().lower(), cls.DEFAULT_COLOR)

    @classmethod
    def draw_detections(
        cls,
        image: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2,
        font_scale: float = 0.55
    ) -> np.ndarray:
        """
        Draw bounding boxes and distance text annotations.
        Format matches: 'Cone, 1.5m', 'Stop Sign, 3.2m', 'Barrier, 2.8m'.
        """
        canvas = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            color = cls.get_color(det.class_name)

            # Draw outer rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

            # Draw subtle corner brackets for a modern robotics HUD look
            c_len = min(15, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
            c_thick = line_thickness + 1
            # Top-left
            cv2.line(canvas, (x1, y1), (x1 + c_len, y1), color, c_thick)
            cv2.line(canvas, (x1, y1), (x1, y1 + c_len), color, c_thick)
            # Top-right
            cv2.line(canvas, (x2, y1), (x2 - c_len, y1), color, c_thick)
            cv2.line(canvas, (x2, y1), (x2, y1 + c_len), color, c_thick)
            # Bottom-left
            cv2.line(canvas, (x1, y2), (x1 + c_len, y2), color, c_thick)
            cv2.line(canvas, (x1, y2), (x1, y2 - c_len), color, c_thick)
            # Bottom-right
            cv2.line(canvas, (x2, y2), (x2 - c_len, y2), color, c_thick)
            cv2.line(canvas, (x2, y2), (x2, y2 - c_len), color, c_thick)

            # Format label: 'Cone, 1.5m'
            label = det.label_text

            # Calculate text size
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Filled label background badge above or inside bbox
            badge_y1 = max(0, y1 - th - baseline - 6)
            badge_y2 = y1 if y1 - th - baseline - 6 >= 0 else y1 + th + baseline + 6
            badge_x2 = min(canvas.shape[1], x1 + tw + 10)

            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Text shadow for maximum readability
            text_y = badge_y2 - baseline - 3
            cv2.putText(
                canvas,
                label,
                (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas

    @classmethod
    def render_dashboard(
        cls,
        main_frame: np.ndarray,
        detections: List[Detection],
        bev_map: Optional[np.ndarray] = None,
        fps: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a robotics perception dashboard with main camera view,
        mini-map BEV inset, and telemetry stats.
        """
        annotated = cls.draw_detections(main_frame, detections)
        h, w = annotated.shape[:2]

        # Top telemetry bar
        bar_height = 36
        bar = np.full((bar_height, w, 3), (20, 20, 20), dtype=np.uint8)
        status_text = "ROBOTICS NAVIGATION PERCEPTION | "
        if model_name:
            status_text += f"Model: {model_name} | "
        if fps:
            status_text += f"FPS: {fps:.1f} | "
        status_text += f"Active Obstacles: {len(detections)}"

        cv2.putText(bar, status_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        dashboard = np.vstack([bar, annotated])

        # Overlay BEV mini-map in top-right corner if provided
        if bev_map is not None:
            inset_size = int(h * 0.40)
            bev_resized = cv2.resize(bev_map, (inset_size, inset_size))
            bev_h, bev_w = bev_resized.shape[:2]

            # Inset position
            pos_x = w - bev_w - 15
            pos_y = bar_height + 15

            # Border
            cv2.rectangle(dashboard, (pos_x - 2, pos_y - 2), (pos_x + bev_w + 2, pos_y + bev_h + 2), (0, 200, 255), 2)
            dashboard[pos_y:pos_y + bev_h, pos_x:pos_x + bev_w] = bev_resized

            # Label on mini-map
            cv2.putText(dashboard, "BEV MAP", (pos_x + 8, pos_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return dashboard
