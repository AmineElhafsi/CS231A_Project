import math

import numpy as np
import torch
import torch.nn.functional as F

def get_exponential_lr_scheduler(
    lr_init: float, 
    lr_final: float, 
    lr_delay_steps: int = 0, 
    lr_delay_mult: float = 1.0, 
    max_steps: int = int(1e6),
):
    """
    Adapted from https://github.com/VladimirYugay/Gaussian-SLAM/blob/main/src/utils/gaussian_model_utils.py
    and Plenoxels.

    Returns exponential decay learning rate scheduler. Initial lr_delay_steps are scaled by a delay_rate
    which is on a reverse cosine decay schedule.
    
    Args:
        lr_init (float): Initial learning rate.
        lr_final (float): Final learning rate.
        lr_delay_steps (int): Number of steps to delay learning rate decay.
        lr_delay_mult (float): Multiplier for delayed learning rate decay.
        max_steps (int): Maximum number of steps.
        
    Returns:
        scheduler (function): Learning rate scheduler.
    """

    def scheduler(step):
        # set learning rate to 0 if step is negative or if initial and final learning rates are 0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        
        # compute learning rate delay multiplier
        if lr_delay_steps > 0:
            p = np.clip(step / lr_delay_steps, 0, 1)
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * p)
        else:
            delay_rate = 1
        
        # compute learning rate
        p = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp((1 - p) * np.log(lr_init) + p * np.log(lr_final))
        return log_lerp * delay_rate
    
    return scheduler


# metrics and loss functions:


def isotropic_loss(scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes the isotropic loss to reduce the emergence of elongated 3D Gaussians.

    Args:
        scaling: The scaling tensors for the 3D Gaussians of shape(N, 3).
    Returns:
        torch.Tensor: The computed isotropic loss.
    """
    mean_scaling = scaling.mean(dim=1, keepdim=True)
    isotropic_diff = torch.abs(scaling - mean_scaling * torch.ones_like(scaling))
    return isotropic_diff.mean()


def l1_loss(prediction: torch.Tensor, target: torch.Tensor, aggregation_method="mean") -> torch.Tensor:
    """
    Computes the L1 loss between a prediction and a target. Optionally specify an aggregation method.

    Args:
        prediction: The predicted tensor.
        target: The ground truth tensor.
        aggregation_method: The aggregation method to be used. Defaults to "mean".
    Returns:
        torch.Tensor: The computed L1 loss.
    """
    l1_loss = torch.abs(prediction - target)
    if aggregation_method == "mean":
        return l1_loss.mean()
    elif aggregation_method == "sum":
        return l1_loss.sum()
    elif aggregation_method == "none":
        return l1_loss
    else:
        raise ValueError("Invalid aggregation method.")
    

def ssim(image_1: torch.Tensor, image_2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        image_1: The first image.
        image_2: The second image.
        window_size: The size of the window for SSIM computation. Defaults to 11.
        size_average: Flag that averages the SSIM over all image pixels if True.
    Returns:
        torch.Tensor: The computed SSIM.
    """
    num_channels = image_1.size(-3) 
    
    # create 2D Gaussian kernel
    sigma = 1.5 # TODO: I don't like that this is hardcoded here
    gaussian = torch.Tensor(
        [math.exp(-1/2 * (x - window_size // 2) ** 2 / float(sigma ** 2)) for x in range(window_size)]
    )
    kernel_1d = (gaussian / gaussian.sum()).unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(num_channels, 1, window_size, window_size).contiguous() # TODO: check if torch.autograd.Variable is needed

    # ensure correct device and type
    if image_1.is_cuda:
        window = window.cuda(image_1.get_device())
    else:
        raise ValueError("SSIM computation requires CUDA.")
    window = window.type_as(image_1)

    # compute ssim
    mu1 = F.conv2d(image_1, window, padding=window_size//2, groups=num_channels)
    mu2 = F.conv2d(image_2, window, padding=window_size//2, groups=num_channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image_1 * image_1, window, padding=window_size//2, groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(image_2 * image_2, window, padding=window_size//2, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(image_1 * image_2, window, padding=window_size//2, groups=num_channels) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_value = ssim_numerator / ssim_denominator

    if size_average:
        return ssim_value.mean()
    else:
        return ssim_value.mean(1).mean(1).mean(1)

# def gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
#     """
#     Supporting function for ssim computations. Creates a 1D Gaussian kernel.

#     Args:
#         window_size: The size of the window.
#         sigma: The standard deviation of the Gaussian.
#     Returns:
#         torch.Tensor: The resulting Gaussian kernel.
#     """
#     gauss = torch.Tensor(
#         [math.exp(-1/2 * (x - window_size // 2) ** 2 / float(sigma ** 2)) for x in range(window_size)]
#     )
#     return gauss / gauss.sum()
