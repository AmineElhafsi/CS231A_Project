import numpy as np
import open3d as o3d
import torch
from torch import nn

from simple_knn._C import distCUDA2 # needs to be imported after torch

from src.utils.math_utils import rgb_to_sh
from src.utils.optimization import get_exponential_lr_scheduler

class GaussianModel:
    def __init__(self, sh_degree: int = 3, isotropic: bool = False) -> None:
        self.gaussian_param_names = [
            "active_sh_degree",
            "xyz",
            "features_dc",
            "features_rest",
            "scaling",
            "rotation",
            "opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "spatial_lr_scale",
            "optimizer",
        ]
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # temp
        self.isotropic = isotropic
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0, 4).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1

        # setup functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize

    def add_new_params_to_optimizer(self, new_parameter_dict):
        # initialize merged (new and existing) parameter dict
        merged_params = {}

        # iterate through optimizer parameter groups and update
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            param_name = group["name"]
            new_params = new_parameter_dict[param_name]

            stored_params = self.optimizer.state.get(group["params"][0], None)
            if stored_params is not None:
                # if the parameter group already exists, initialize optimizer state for new parameters
                stored_params["exp_avg"] = torch.cat(
                    (stored_params["exp_avg"], torch.zeros_like(new_params)), dim=0
                )
                stored_params["exp_avg_sq"] = torch.cat(
                    (stored_params["exp_avg_sq"], torch.zeros_like(new_params)), dim=0
                )

                # delete and replace existing optimizer state
                del self.optimizer.state[group["params"][0]] #TODO: check if this is needed still - possibly prevents memory leaks.
                group["params"][0] = nn.Parameter( # add new parameters to parameter group
                    torch.cat((group["params"][0], new_params), dim=0).requires_grad_(True)
                )                
                self.optimizer.state[group["params"][0]] = stored_params
                
            else:
                # add new parameters
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], new_params), dim=0).requires_grad_(True)
                )
            
            # add new parameter set to dictionary
            merged_params[param_name] = group["params"][0]
        
        return merged_params
    
    def remove_parameters_from_optimizer(self, remove_mask):
        # initialize pruned parameter dict (output dict)
        pruned_params = {}
        keep_mask = ~remove_mask

        # iterate through optimizer parameter groups and update
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1 # TODO: maybe this should be removed

            # remove optimizer state for removed parameters
            stored_params = self.optimizer.state.get(group["params"][0], None)
            if stored_params is not None:
                stored_params["exp_avg"] = stored_params["exp_avg"][keep_mask]
                stored_params["exp_avg_sq"] = stored_params["exp_avg_sq"][keep_mask]

                # delete and replace existing optimizer state
                del self.optimizer.state[group["params"][0]] #TODO: check if this is needed still - possibly prevents memory leaks.
                group["params"][0] = nn.Parameter((group["params"][0][keep_mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_params
            else:
                # remove parameters
                group["params"][0] = nn.Parameter(group["params"][0][keep_mask].requires_grad_(True))
            # add pruned parameter set to dictionary
            pruned_params[group["name"]] = group["params"][0]
        
        return pruned_params

    def add_points(self, point_cloud: o3d.geometry.PointCloud, global_scale_init=True):
        """
        Adds a point cloud to the Gaussian model.
        Args:
            point_cloud (o3d.geometry.PointCloud): Point cloud to add.
            global_scale_init (bool): If True, initializes the scale of the new points to the global scale.
        """
        # convert point cloud to torch
        fused_point_cloud = torch.tensor(np.asarray(point_cloud.points)).float().cuda()

        # set features
        fused_color = rgb_to_sh(torch.tensor(np.asarray(point_cloud.colors)).float().cuda())
        features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda())
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # scale initialization for Gaussian points based on mean distance to neighboring Gaussian points
        # if global_scale_init is True, neighbors are all particles; otherwise, only the new point cloud points are considered
        if global_scale_init:
            global_points = torch.cat(
                (
                    self.get_xyz(), 
                    torch.from_numpy(np.asarray(point_cloud.points)).float().cuda()
                )
            )
            dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001)
            dist2 = dist2[self.get_size():]
        else:
            point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points)).float().cuda()
            dist2 = torch.clamp_min(distCUDA2(point_cloud_points), 0.0000001)
        scale_factor = 1.0
        scales = torch.log(scale_factor * torch.sqrt(dist2))[..., None].repeat(1, 3)

        # rotation initialization (quaternion representation with w as the first element)
        rotations = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rotations[:, 0] = 1.0

        # opacity initialization (uniformly initialized to 0.5 for all points)
        opacities = self.inverse_opacity_activation(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        # add new points as model parameters
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rotations.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        # update model: give new parameters to optimizer and add new parameters to model
        self.update_model(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def update_model(self, new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity):
        new_parameter_dict = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
        }

        # add new parameters to optimizer and get the new collection of merged parameters
        merged_parameters = self.add_new_params_to_optimizer(new_parameter_dict)

        # update Gaussian model attributes with merged parameters
        self._xyz = merged_parameters["xyz"]
        self._features_dc = merged_parameters["f_dc"]
        self._features_rest = merged_parameters["f_rest"]
        self._scaling = merged_parameters["scaling"]
        self._rotation = merged_parameters["rotation"]
        self._opacity = merged_parameters["opacity"]

        # TODO: check these shapes
        self.xyz_gradient_accum = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz().shape[0]), device="cuda")


    def capture_dict(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.clone().detach().cpu(),
            "features_dc": self._features_dc.clone().detach().cpu(),
            "features_rest": self._features_rest.clone().detach().cpu(),
            "scaling": self._scaling.clone().detach().cpu(),
            "rotation": self._rotation.clone().detach().cpu(),
            "opacity": self._opacity.clone().detach().cpu(),
            "max_radii2D": self.max_radii2D.clone().detach().cpu(),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone().detach().cpu(),
            "denom": self.denom.clone().detach().cpu(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict(),
        }
    
    def get_size(self):
        return self._xyz.shape[0]

    def get_xyz(self):
        return self._xyz
    
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_scaling(self):
        if self.isotropic:
            scale = self.scaling_activation(self._scaling)[:, 0:1]
            scales = scale.repeat(1, 3)
            return scales
        else:
            return self.scaling_activation(self._scaling)
    
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def remove_points(self, remove_mask):
        # remove corresponding parameters from optimizer
        pruned_parameters = self.remove_parameters_from_optimizer(remove_mask)

        # update model with pruned points
        self._xyz = pruned_parameters["xyz"]
        self._features_dc = pruned_parameters["f_dc"]
        self._features_rest = pruned_parameters["f_rest"]
        self._scaling = pruned_parameters["scaling"]
        self._rotation = pruned_parameters["rotation"]
        self._opacity = pruned_parameters["opacity"]

        keep_mask = ~remove_mask
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        self.max_radii2D = self.max_radii2D[keep_mask]       
        
        
    def training_setup(self, optimization_params):
        self.percent_dense = optimization_params.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz().shape[0], 1), device="cuda") # what is this?
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda") # what is this?

        # define optimization parameter groups and associated optimizer arguments
        params = [
            {"params": [self._xyz], "lr": optimization_params.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": optimization_params.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": optimization_params.feature_lr/20., "name": "f_rest"},
            {"params": [self._opacity], "lr": optimization_params.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": optimization_params.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": optimization_params.rotation_lr, "name": "rotation"},
        ]

        # create optimizer
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        # learning rate scheduler
        self.xyz_scheduler = get_exponential_lr_scheduler(
            lr_init=optimization_params.position_lr_init * self.spatial_lr_scale,
            lr_final=optimization_params.position_lr_final * self.spatial_lr_scale,
            # lr_delay_mult=optimization_params.position_lr_delay_mult, # doesn't do anything since lr_delay_steps is 0
            max_steps=optimization_params.position_lr_max_steps,
        )
    
    