"""
Evaluation Metrics for Object Detection and Distance Estimation.
"""
from typing import List, Dict
import numpy as np


def compute_distance_metrics(ground_truth_dists: List[float], estimated_dists: List[float]) -> Dict[str, float]:
    """
    Compute error metrics between estimated distances and ground truth distances.
    - Mean Absolute Error (MAE) in meters
    - Root Mean Squared Error (RMSE) in meters
    - Mean Absolute Percentage Error (MAPE)
    - Max Error in meters
    """
    gt = np.array(ground_truth_dists, dtype=np.float64)
    est = np.array(estimated_dists, dtype=np.float64)

    errors = np.abs(est - gt)
    rel_errors = errors / (gt + 1e-6) * 100.0

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((est - gt)**2)))
    mape = float(np.mean(rel_errors))
    max_err = float(np.max(errors))

    return {
        "mae_meters": mae,
        "rmse_meters": rmse,
        "mape_percent": mape,
        "max_error_meters": max_err,
        "sample_count": len(gt)
    }
