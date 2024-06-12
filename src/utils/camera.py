import math

import numpy as np
import torch


def get_camera_intrinsics_matrix(config: dict) -> np.ndarray:
    """
    Returns the camera intrinsics matrix from a configuration dictionary.
    Args:
        config: A dictionary containing the camera intrinsics parameters.
    Returns:
        K: The camera intrinsics matrix.
    """
    if config["mode"] == "dataset":
        fx, fy = config["dataset_config"]["fx"], config["dataset_config"]["fy"]
        cx, cy = config["dataset_config"]["cx"], config["dataset_config"]["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    return K


def rotation_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to Euler angles.
    Args:
        R: A rotation matrix.
    Returns:
        Euler angles corresponding to the rotation matrix.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z]) * (180 / math.pi)


def exceeds_motion_thresholds(
    Tc2w_current: torch.Tensor,
    Tc2w_reference: torch.Tensor,
    translation_threshold: float = 0.5,
    rotation_threshold: float = 50.0,
) -> bool:
    """
    Checks if the current pose exceeds the rotation and translation thresholds from a 
    reference pose. 
    Args:
        Tc2w_current (torch.Tensor): Current camera-to-world transform.
        Tc2w_reference (torch.Tensor): Reference camera-to-world transform.
        translation_threshold (float): Translation threshold in meters.
        rotation_threshold (float): Rotation threshold in degrees.
    Returns:
        exceeds_thresholds: A boolean indicator of whether the pose difference exceeds the specified 
        translation or rotation thresholds.
    """
    Tw2c_reference = torch.linalg.inv(Tc2w_reference).float()
    T_diff = Tw2c_reference @ Tc2w_current # T_diff transforms current camera to reference camera

    translated_distance = torch.norm(T_diff[:3, 3])
    rotated_distance = torch.abs(rotation_to_euler(T_diff[:3, :3]))

    translation_exceeded = (translated_distance > translation_threshold)
    rotation_exceeded = torch.any(rotated_distance > rotation_threshold)
    result = (translation_exceeded or rotation_exceeded).item()
    return result
