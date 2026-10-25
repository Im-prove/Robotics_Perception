"""
YOLO Object Detector Implementation supporting PyTorch and ONNX Runtime backends.
"""
from typing import List, Tuple, Optional, Dict, Any
import os
import time
import numpy as np
import cv2
import torch
from .base_detector import BaseDetector, Detection
from ..distance_estimation.fusion_estimator import FusedDistanceEstimator
from ..distance_estimation.geometry import CameraIntrinsics, CameraExtrinsics


class YOLODetector(BaseDetector):
    """
    Modular YOLO detector supporting PyTorch and ONNX models with integrated distance estimation.
    """

    TARGET_CLASSES = {
        "cone": ["cone", "traffic cone", "traffic_cone", "fire hydrant"],
        "barrier": ["barrier", "traffic barrier", "construction barrier", "fence"],
        "stop_sign": ["stop sign", "stop_sign", "traffic sign"]
    }

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cpu",
        distance_estimator: Optional[FusedDistanceEstimator] = None,
        custom_class_map: Optional[Dict[int, str]] = None
    ):
        self.weights_path = weights_path
        self.device = device
        self.distance_estimator = distance_estimator
        self.is_onnx = weights_path.endswith(".onnx")
        self.custom_class_map = custom_class_map or {}

        if self.is_onnx:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(weights_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.model = None
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.model.to(device)
            self.session = None

    def _normalize_class_name(self, raw_name: str) -> Optional[str]:
        """Map raw detected class name to standard assignment classes: Cone, Barrier, Stop Sign."""
        raw = raw_name.strip().lower()

        for canonical, aliases in self.TARGET_CLASSES.items():
            if raw in aliases or any(a in raw for a in aliases):
                if canonical == "cone":
                    return "Cone"
                elif canonical == "barrier":
                    return "Barrier"
                elif canonical == "stop_sign":
                    return "Stop Sign"
        return None

    def predict(
        self,
        image: np.ndarray,
        conf_thresh: float = 0.35,
        filter_target_only: bool = True
    ) -> List[Detection]:
        """Run detection and compute distance estimation for each obstacle."""
        if self.is_onnx:
            return self._predict_onnx(image, conf_thresh, filter_target_only)
        else:
            return self._predict_torch(image, conf_thresh, filter_target_only)

    def _predict_torch(
        self,
        image: np.ndarray,
        conf_thresh: float,
        filter_target_only: bool
    ) -> List[Detection]:
        results = self.model.predict(image, conf=conf_thresh, verbose=False)
        detections: List[Detection] = []

        if not results:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        names = self.model.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            raw_name = self.custom_class_map.get(cls_id, names.get(cls_id, f"class_{cls_id}"))
            norm_name = self._normalize_class_name(raw_name)

            if filter_target_only and norm_name is None:
                continue

            display_name = norm_name if norm_name is not None else raw_name

            det = Detection(
                class_id=cls_id,
                class_name=display_name,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2)
            )

            # Estimate distance
            if self.distance_estimator is not None:
                dist_res = self.distance_estimator.estimate(display_name, (x1, y1, x2, y2))
                if dist_res is not None:
                    det.distance_m = dist_res.distance_m
                    det.distance_z_m = dist_res.distance_z_m
                    det.offset_x_m = dist_res.offset_x_m

            detections.append(det)

        return detections

    def _predict_onnx(
        self,
        image: np.ndarray,
        conf_thresh: float,
        filter_target_only: bool
    ) -> List[Detection]:
        """ONNX Runtime optimized edge inference."""
        h_orig, w_orig = image.shape[:2]
        input_size = 640

        # Preprocessing: resize with letterbox
        img_resized = cv2.resize(image, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) if len(img_resized.shape) == 3 else img_resized
        blob = img_rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self.session.run(self.output_names, {self.input_name: blob})
        output = outputs[0]  # shape: [1, 84, 8400]

        # YOLOv8 format: [cx, cy, w, h, class_probs...]
        if output.shape[1] < output.shape[2]:
            output = np.transpose(output[0], (1, 0))  # [8400, 84]
        else:
            output = output[0]

        boxes = []
        confidences = []
        class_ids = []

        scale_x = w_orig / input_size
        scale_y = h_orig / input_size

        for pred in output:
            scores = pred[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf > conf_thresh:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                x1 = (cx - w / 2) * scale_x
                y1 = (cy - h / 2) * scale_y
                x2 = (cx + w / 2) * scale_x
                y2 = (cy + h / 2) * scale_y
                boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                confidences.append(conf)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
        detections: List[Detection] = []

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                x1, y1, x2, y2 = x, y, x + w, y + h
                cls_id = class_ids[idx]
                conf = confidences[idx]

                raw_name = self.custom_class_map.get(cls_id, f"class_{cls_id}")
                norm_name = self._normalize_class_name(raw_name)

                if filter_target_only and norm_name is None:
                    continue

                display_name = norm_name if norm_name is not None else raw_name
                det = Detection(
                    class_id=cls_id,
                    class_name=display_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2)
                )

                if self.distance_estimator is not None:
                    dist_res = self.distance_estimator.estimate(display_name, (x1, y1, x2, y2))
                    if dist_res is not None:
                        det.distance_m = dist_res.distance_m
                        det.distance_z_m = dist_res.distance_z_m
                        det.offset_x_m = dist_res.offset_x_m

                detections.append(det)

        return detections

    def get_model_size_mb(self) -> float:
        """Return file size of weights in MB."""
        if os.path.exists(self.weights_path):
            return os.path.getsize(self.weights_path) / (1024.0 * 1024.0)
        return 0.0
