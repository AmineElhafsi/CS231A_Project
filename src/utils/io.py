from pathlib import Path
from typing import Union

import torch
import yaml


def load_yaml(file_path: Union[str, Path]) -> dict:
    """ Loads a YAML file and returns its contents as a dictionary.
    Args:
        file_path: The path to the YAML file.
    Returns:
        The dictionary containing the contents of the YAML file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def mkdir_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_dict_to_ckpt(dictionary: dict, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a checkpoint file in the specified directory, creating the directory 
    if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the checkpoint file.
        directory: The directory where the checkpoint file will be saved.
    """
    torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)
    

@mkdir_decorator
def save_dict_to_yaml(dictionary: dict, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a YAML file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the YAML file.
        directory: The directory where the YAML file will be saved.
    """
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)