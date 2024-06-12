import os
from datetime import datetime
from pathlib import Path

import torch

from src.splatting.datasets import get_dataset
from src.splatting.gaussian_model import GaussianModel
from src.splatting.data_logging import Logger
from src.splatting.mapper import Mapper
from src.splatting.parameters import OptimizationParams
from src.utils.camera import exceeds_motion_thresholds
from src.utils.io import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.utils import setup_seed


class GaussianSplatting:
    def __init__(self, config: dict) -> None:
        # prepare output
        self._setup_output_path(config)
        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        # parse configuration
        self.config = config
        self.mode = config["mode"]
        self.submap_creation_criterion = config["mapping"]["submap_creation_criteria"]["criterion"]

        # prepare logging
        self.logger = Logger(self.output_path, use_wandb=False)

        # set operating mode
        self.dataset = None
        if self.mode == "dataset": 
            self.dataset = get_dataset(config["dataset_config"]["dataset"])(config["dataset_config"])
        elif self.mode == "streaming":
            raise NotImplementedError("Streaming mode not implemented.")
        
        # initialize model
        setup_seed(self.config["seed"])
        self.opt = OptimizationParams()
        self.gaussian_model = GaussianModel(0)
        self.gaussian_model.training_setup(self.opt)

        # prepare frame and submap accounting
        self.frame_id = 0
        self.submap_id = 0
        self.new_submap_frame_ids = [0]
        self.keyframes_info = {}

        # mapping module
        self.mapper = Mapper(config["mapper"], self.logger)

    def _setup_output_path(self, config: dict) -> None:
        """ 
        Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.

        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        self.output_path = Path(config["output_path"])
        data_name = config["dataset_config"]["dataset_path"].split("/")[-1] 
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.output_path = self.output_path / self.timestamp
        self.output_path = self.output_path / data_name
        self.output_path.mkdir(exist_ok=True, parents=True)
        os.makedirs(self.output_path / "mapping", exist_ok=True)

    def should_start_new_submap(self):
        if self.submap_creation_criterion == "motion_threshold":
            current_T_c2w = self.T_c2ws[self.frame_id]
            submap_T_c2w = self.T_c2ws[self.new_submap_frame_ids[-1]]
            start_new_map = exceeds_motion_thresholds(
                current_T_c2w, 
                submap_T_c2w, 
                self.config["mapping"]["submap_creation_criteria"]["translation_threshold"], 
                self.config["mapping"]["submap_creation_criteria"]["rotation_threshold"]
            )
            return start_new_map
        else:
            raise NotImplementedError(f"Criterion {self.submap_creation_criterion} not implemented.")
        

    def start_new_submap(self) -> None:
        """ 
        Initializes a new submap, saving the current submap's checkpoint and resetting the Gaussian model.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        """
        # save current submap and camera trajectory
        gaussian_params = self.gaussian_model.capture_dict()
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys())),
        }
        save_dict_to_ckpt(
            submap_ckpt,
            f"submap_{self.submap_id}".zfill(6) + ".ckpt",
            directory=(self.output_path/"submaps")
        )
        save_dict_to_ckpt(
            self.T_c2ws[:self.frame_id + 1], 
            "T_c2w.ckpt", 
            directory=self.output_path
        )

        # reset Gaussian model and keyframe info
        self.gaussian_model = GaussianModel(0)
        self.gaussian_model.training_setup(self.opt) # TODO: possibly redundant
        self.mapper.keyframes = []
        self.keyframes_info = {}

        # update submap tracking (TODO: check motion heuristic)
        self.new_submap_frame_ids.append(self.frame_id)
        # TODO: check need for this: self.mapping_frame_ids.append(self.frame_id)
        self.submap_id += 1








    def step(self, keyframe: dict) -> None:
        # check if new submap is needed:
        if self.should_start_new_submap():
            print("Starting new submap")
            self.start_new_submap()

        # map frame
        # TODO: check if this if is needed
        # if self.frame_id in self.mapping_frame_ids:
        print(f"Mapping frame {self.frame_id}")
        self.gaussian_model.training_setup(self.opt)
        is_new_submap = not bool(self.keyframes_info)
        results_dict = self.mapper.map(
            keyframe,
            self.gaussian_model,
            is_new_submap
        )

        # Keyframes info update
        self.keyframes_info[keyframe["frame_id"]] = {
            "keyframe_id": len(self.keyframes_info.keys()),
            "opt_dict": results_dict
        }

        frame_id = keyframe["frame_id"] 
        save_dict_to_ckpt(keyframe, f"keyframe_{frame_id}.ckpt", directory=(self.output_path/"keyframes"))
        save_dict_to_ckpt(results_dict, f"results_{frame_id}.ckpt", directory=(self.output_path/"results"))


            
        

    def run_from_dataset(self) -> None:
        dataset = self.dataset
        self.T_c2ws = torch.zeros(len(dataset), 4, 4)

        # iterate over dataset
        for frame_id in range(len(dataset)):
            self.frame_id = frame_id

            # assume pose is ground truth (assume robot has other means of state estimation)
            keyframe = {
                "frame_id": frame_id,
                "color": dataset[frame_id][1],
                "depth": dataset[frame_id][2],
                "T_c2w": dataset[frame_id][3],
                "H": dataset.image_height,
                "W": dataset.image_width,
                "K": dataset.K, # camera intrinsics matrix
            }
            # image, T_c2w = dataset[frame_id][0], dataset[frame_id][-1]
            self.T_c2ws[frame_id] = torch.from_numpy(keyframe["T_c2w"])

            self.step(keyframe)

        save_dict_to_ckpt(self.T_c2ws[:frame_id + 1], "T_c2w.ckpt", directory=self.output_path)



if __name__ == "__main__":
    import yaml

    # load yaml file config
    with open("/home/amine/Documents/Research/gaussian-splatting-playground/config/test.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Create an instance of GaussianSplatting with the loaded config
    splatting = GaussianSplatting(config)
    
    # Call the run_from_dataset method with the desired dataset
    splatting.run_from_dataset()