from typing import Any, Dict, List


def summarize_force_map(force_map: Any) -> Dict[str, float]:
    """
    Return min/max/mean summary for a force map.

    Accepts either a numpy array or a nested Python list.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("numpy is required for force map utilities.") from exc

    array = np.asarray(force_map, dtype=float)

    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def normalize_force_map(force_map: Any) -> List[List[float]]:
    """
    Normalize a force map to [0, 1] for visualization.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("numpy is required for force map utilities.") from exc

    array = np.asarray(force_map, dtype=float)
    min_value = array.min()
    max_value = array.max()

    if max_value - min_value < 1e-9:
        return np.zeros_like(array).tolist()

    normalized = (array - min_value) / (max_value - min_value)
    return normalized.tolist()
