import os
import random

import numpy as np
import open3d
import torch


def setup_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility.
    Args:
        seed (int): Seed value for torch, numpy, and random.
    """
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.
    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        np.ndarray: NumPy array.
    """
    return tensor.detach().cpu().numpy()


def numpy_to_point_cloud(points: np.ndarray, rgb=None) -> open3d.geometry.PointCloud:
    """
    Converts NumPy array to o3d point cloud.

    Args:
        points (ndarray): Point cloud as an array.
    Returns:
        (PointCloud): PointCloud in o3d format.
    """
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)
    if rgb is not None:
        cloud.colors = open3d.utility.Vector3dVector(rgb)
    return cloud


def numpy_to_torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor.
    Args:
        array (np.ndarray): Input array.
        device (str): Device to store the tensor.
    Returns:
        torch.Tensor: PyTorch tensor.
    """
    return torch.from_numpy(array).float().to(device)

if __name__ == "__main__":
    setup_seed(42)
    print("Seed is set.")