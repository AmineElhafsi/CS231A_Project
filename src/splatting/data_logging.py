from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


class Logger:
    def __init__(self, output_path: Union[Path, str], use_wandb: bool = False) -> None:
        self.output_path = Path(output_path)
        (self.output_path / "mapping").mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb

    def log_iteration(self, frame_id, new_pts_num, model_size, iter_opt_time, results_dict: dict) -> None:
        """
        Logs mapping iteration metrics including the number of new points, model size, and optimization times,
        and optionally reports to Weights & Biases (wandb).

        Args:
            frame_id: Identifier for the current frame.
            new_pts_num: The number of new points added in the current mapping iteration.
            model_size: The total size of the model after the current mapping iteration.
            iter_opt_time: Time taken per optimization iteration.
            opt_dict: A dictionary containing optimization metrics such as PSNR, color loss, and depth loss.
        """
        if self.use_wandb:
            wandb.log({"Mapping/idx": frame_id,
                       "Mapping/num_total_gs": model_size,
                       "Mapping/num_new_gs": new_pts_num,
                       "Mapping/per_iteration_time": iter_opt_time,
                       "Mapping/psnr_render": results_dict["psnr_render"],
                       "Mapping/color_loss": results_dict[frame_id]["color_loss"],
                       "Mapping/depth_loss": results_dict[frame_id]["depth_loss"]})
            
    def vis_mapping_iteration(self, frame_id, iter, color, depth, gt_color, gt_depth, seeding_mask=None) -> None:
        """
        Visualization of depth, color images and save to file.

        Args:
            frame_id (int): current frame index.
            iter (int): the iteration number.
            save_rendered_image (bool): whether to save the rgb image in separate folder
            img_dir (str): the directory to save the visualization.
            seeding_mask: used in mapper when adding gaussians, if not none.
        """
        gt_depth_np = gt_depth.cpu().numpy()
        gt_color_np = gt_color.cpu().numpy()

        depth_np = depth.detach().cpu().numpy()
        color = torch.round(color * 255.0) / 255.0
        color_np = color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # make errors >=5cm noticeable
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[np.squeeze(gt_depth_np == 0.0)] = 0.0

        # Determine Aspect Ratio and Figure Size
        aspect_ratio = color.shape[1] / color.shape[0]
        fig_height = 8
        # Adjust the multiplier as needed for better spacing
        fig_width = fig_height * aspect_ratio * 1.2

        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 0].set_title('Input Depth', fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title('Rendered Depth', fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma")
        axs[0, 2].set_title('Depth Residual', fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB', fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Rendered RGB', fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        if seeding_mask is not None:
            axs[1, 2].imshow(seeding_mask, cmap="gray")
            axs[1, 2].set_title('Densification Mask', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
        else:
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

        for ax in axs.flatten():
            ax.axis('off')
        fig.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin
        fig_name = str(self.output_path / "mapping" / f'{frame_id:04d}_{iter:04d}.jpg')
        fig_title = f"Mapper Color/Depth at frame {frame_id:04d} iters {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)
        plt.savefig(fig_name, dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            log_title = "Mapping_vis/" + f'{frame_id:04d}_{iter:04d}'
            wandb.log({log_title: [wandb.Image(fig_name)]})
        print(f"Saved rendering vis of color/depth at {frame_id:04d}_{iter:04d}.jpg")