import yaml

from src.splatting.gaussian_splatting import GaussianSplatting

if __name__ == "__main__":

    dataset_path_list = [
        # replica datasets
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_office0.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_office1.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_office2.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_office3.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_office4.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_room0.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_room1.yaml",
        # "/home/amine/Documents/Research/gaussian-splatting-playground/config/replica_room2.yaml",

        # TUM datasets
        "/home/amine/Documents/Research/gaussian-splatting-playground/config/tum_freiburg1_desk.yaml",
        "/home/amine/Documents/Research/gaussian-splatting-playground/config/tum_freiburg2_xyz.yaml",
        "/home/amine/Documents/Research/gaussian-splatting-playground/config/tum_freiburg3_long_office_household.yaml",
    ]

    for dataset_path in dataset_path_list:
        print("Processing dataset: ", dataset_path)
        # load yaml file config
        with open(dataset_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Create an instance of GaussianSplatting with the loaded config
        splatting = GaussianSplatting(config)
        
        # Call the run_from_dataset method with the desired dataset
        splatting.run_from_dataset()