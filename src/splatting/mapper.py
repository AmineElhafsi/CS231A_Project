import time

import numpy as np
import torch
import torchvision

from src.splatting.gaussian_model import GaussianModel
from src.splatting.data_logging import Logger
from src.splatting.parameters import OptimizationParams
from src.utils.evaluation import psnr
from src.utils.optimization import isotropic_loss, l1_loss, ssim
from src.utils.rendering import get_render_settings, render_gaussian_model
from src.utils.splatting import (compute_camera_frustum_corners, compute_frustum_point_ids, 
                                 create_point_cloud, geometric_edge_mask, keyframe_optimization_sampling_distribution, 
                                 sample_pixels_based_on_gradient, select_new_point_ids)
from src.utils.utils import numpy_to_torch, numpy_to_point_cloud, torch_to_numpy


class Mapper():
    def __init__(self, config: dict, logger: Logger) -> None:
        # parse configuration
        self.config = config
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_threshold = config["alpha_threshold"]
        self.pruning_threshold = config["pruning_threshold"]
        self.current_view_weight = config["current_view_weight"]

        self.opt = OptimizationParams()

        self.logger = logger

        self.keyframes = []

    def compute_seeding_mask(self, keyframe: dict, gaussian_model: GaussianModel, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded.
        Seeding is based on color gradient for new submaps and alpha masks and depth error for existing submasks.

        Args:
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            gaussian_model: The current submap
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap:
            color_image = keyframe["color"]
            seeding_mask = geometric_edge_mask(color_image, RGB=True)
        else:
            # TODO: check this again
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_threshold)
            gt_depth_tensor = numpy_to_torch(keyframe["depth"], device="cuda")[None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch_to_numpy(seeding_mask[0])

        return seeding_mask
    
    def seed_new_gaussians(self, keyframe: dict, seeding_mask: np.ndarray, new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on keyframe, a seeding mask, and a flag indicating whether this is a new submap.

        Args:
            keyframe: A dictionary with the current frame_id, gt_rgb, gt_depth, camera-to-world transform and image information.
                color: The ground truth color image as a numpy array with shape (H, W, 3).
                depth: The ground truth depth map as a numpy array with shape (H, W).
                K: The camera intrinsics matrix as a numpy array with shape (3, 3).
                T_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 6) (Last dimension containts xyzrgb values)
        """
        pc_points = create_point_cloud(keyframe["color"], 1.005 * keyframe["depth"], keyframe["K"], keyframe["T_c2w"])

        flat_depth_mask = (keyframe["depth"] > 0.).flatten()
        valid_ids = np.flatnonzero(seeding_mask)
        if new_submap:
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pc_points.shape[0])
            else:
                assert self.new_submap_points_num <= pc_points.shape[0] # don't sample more points than pixels
                uniform_ids = np.random.choice(pc_points.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(keyframe["color"], self.new_submap_gradient_points_num)
            sample_ids = np.unique(np.concatenate((uniform_ids, gradient_ids, valid_ids)))
        else:
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[flat_depth_mask[sample_ids]]
        points = pc_points[sample_ids, :].astype(np.float32)
        
        return points
    
    def grow_submap(self, keyframe: dict, seeded_points: np.ndarray, gaussian_model: GaussianModel, filter_cloud: bool) -> int:
        """
        Grows the current submap by adding new Gaussians from the current keyframe.

        Args:
            keyframe: A dictionary with the current frame_id, gt_rgb, gt_depth, camera-to-world transform and image information.
            seeded_points: An array of 3D points where new Gaussians will be initialized, with shape (N, 6) (xzyrgb).
            gaussian_model (GaussianModel): The current Gaussian model of the submap.
            filter_cloud: A boolean flag indicating whether to filter the point cloud for outliers/noise before adding to the submap.
        Returns:
            int: The number of new points added to the submap.
        """
        # get existing points in submap
        gaussian_points = gaussian_model.get_xyz()

        # determine subset of existing points within camera frustum
        camera_frustum_corners = compute_camera_frustum_corners(keyframe["depth"], keyframe["K"], keyframe["T_c2w"])
        existing_point_ids = compute_frustum_point_ids(gaussian_points, numpy_to_torch(camera_frustum_corners), device="cuda")

        # select new points to add to submap based on density of existing submaps points
        new_point_ids = select_new_point_ids(
            gaussian_points[existing_point_ids], 
            numpy_to_torch(seeded_points[:, :3]).contiguous(), # slice removes rgb values and takes points' xyz data only
            radius=self.new_points_radius,
            device="cuda"
        )
       
        new_point_ids = torch_to_numpy(new_point_ids)

        # add points
        if new_point_ids.shape[0] > 0:
            cloud_to_add = numpy_to_point_cloud(seeded_points[new_point_ids, :3], seeded_points[new_point_ids, 3:] / 255.0)
            if filter_cloud:
                cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            gaussian_model.add_points(cloud_to_add)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        print("Gaussian model size", gaussian_model.get_size())
        
        return new_point_ids.shape[0]
    
    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.

        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """
        iteration = 0
        results_dict = {}
        num_keyframes = len(keyframes)
        
        # get view optimization distribution (how optimization iterations are distributed among keyframes)
        current_frame_iterations = self.current_view_weight * iterations
        view_distribution = keyframe_optimization_sampling_distribution(num_keyframes, iterations, current_frame_iterations)

        start_time = time.time()

        while iteration < iterations + 1:
            # print("gaussian_model xyz: ", gaussian_model.get_xyz().shape)
            # initialize optimizer
            gaussian_model.optimizer.zero_grad(set_to_none=True)

            # sample view
            sampled_id = np.random.choice(np.arange(num_keyframes), p=view_distribution)
            keyframe = keyframes[sampled_id]
            frame_id = keyframe["frame_id"] # TODO: need a better solution for frame id management

            # render model
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            rendered_image, rendered_depth = render_dict["color"], render_dict["depth"]
            gt_image = keyframe["color_torch"]
            gt_depth = keyframe["depth_torch"] # TODO: converted keyframe elements to tensor

            # mask out invalid depth values
            mask = (gt_depth > 0) & (~torch.isnan(rendered_depth)).squeeze(0)

            # compute depth loss
            depth_loss = l1_loss(rendered_depth[:, mask], gt_depth[mask])

            # compute color loss
            weight = self.opt.lambda_dssim
            pixelwise_color_loss = l1_loss(rendered_image[:, mask], gt_image[:, mask])
            ssim_loss = (1.0 - ssim(rendered_image, gt_image)) # TODO: check why mask isn't used here
            color_loss = (1.0 - weight) * pixelwise_color_loss + weight * ssim_loss

            # compute isotropic regularization loss
            isotropic_regularization_loss = isotropic_loss(gaussian_model.get_scaling())

            # compute total loss (assume uniform weighting across all terms)
            total_loss = color_loss + depth_loss + isotropic_regularization_loss
            results_dict[frame_id] = {
                "depth_loss": depth_loss.item(),
                "color_loss": color_loss.item(),
                "isotropic_loss": isotropic_regularization_loss.item(),
                "total_loss": total_loss.item()
            } # TODO: this isn't perfect logging, fix to keep track of all statistics

            # print("depth_loss: ", depth_loss.item())
            # print("color_loss: ", color_loss.item())
            # print("isotropic_loss: ", isotropic_regularization_loss.item())
            # print("total_loss: ", total_loss.item())


            # backpropagate
            total_loss.backward()

            with torch.no_grad():
                # check halfway and at end of optimization for points to remove based on opacity
                if iteration == iterations // 2 or iteration == iterations:
                    remove_mask = (gaussian_model.get_opacity() < self.pruning_threshold).squeeze()
                    gaussian_model.remove_points(remove_mask)

                # optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True) # TODO: this probably isn't needed again
            
            iteration += 1
            # if iteration == 100:
            #     breakpoint()
        
        # log optimization statistics
        torch.cuda.synchronize()
        optimization_time = time.time() - start_time
        results_dict["total_optimization_time"] = optimization_time
        results_dict["average_optimization_iteration_time"] = optimization_time / iterations
        results_dict["num_gaussians"] = gaussian_model.get_size()
        return results_dict

    def map(self, keyframe: dict, gaussian_model: GaussianModel, new_submap: bool) -> dict:
        """
        Mapping iteration that seeds new Gaussians, adds them to the submap, and then optimizes the submap.

        Args:
            keyframe_dict: A dictionary with the current frame_id, gt_rgb, gt_depth, camera-to-world transform and image information.
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """
        # assemble keyframe
        T_w2c = np.linalg.inv(keyframe["T_c2w"])
        keyframe["render_settings"] = get_render_settings(
            keyframe["H"],
            keyframe["W"],
            keyframe["K"],
            T_w2c
        )

        color_transform = torchvision.transforms.ToTensor()
        keyframe["color_torch"] = color_transform(keyframe["color"]).cuda()
        keyframe["depth_torch"] = numpy_to_torch(keyframe["depth"], device="cuda") # TODO: converted keyframe elements to tensor

        # seed Gaussians
        seeding_mask = self.compute_seeding_mask(keyframe, gaussian_model, new_submap)
        seeded_points = self.seed_new_gaussians(keyframe, seeding_mask, new_submap)

        # add points to map
        filter_cloud = False # not new_submap # TODO: tune this
        new_pts_num = self.grow_submap(keyframe, seeded_points, gaussian_model, filter_cloud)

        # optimize submap
        max_iterations = self.new_submap_iterations if new_submap else self.iterations
        start_time = time.time()
        results_dict = self.optimize_submap([keyframe] + self.keyframes, gaussian_model, max_iterations)
        results_dict["new_submap"] = new_submap
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)

        # append keyframe to list
        self.keyframes.append(keyframe)

        # visualizations and logging
        with torch.no_grad():
            render = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            rendered_image, rendered_depth = render["color"], render["depth"]
            psnr_value = psnr(rendered_image, keyframe["color_torch"]).mean().item()
            ssim_value = ssim(rendered_image, keyframe["color_torch"]).item()
            results_dict["psnr_render"] = psnr_value
            results_dict["ssim_render"] = ssim_value
            print(f"PSNR this frame: {psnr_value}")
            print(f"SSIM this frame: {ssim_value}")
            self.logger.vis_mapping_iteration(
                keyframe["frame_id"], max_iterations,
                rendered_image.clone().detach().permute(1, 2, 0),
                rendered_depth.clone().detach().permute(1, 2, 0),
                keyframe["color_torch"].permute(1, 2, 0),
                keyframe["depth_torch"].unsqueeze(-1),
                seeding_mask=seeding_mask
            )

        return results_dict

        
    



        
