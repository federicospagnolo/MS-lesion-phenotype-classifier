import copy

from typing import Any, Union, Dict, List
import numpy as np
import torch


def make_list(param, len_to_have: int, name: str) -> List:
    """
    Ensure that a parameter is a list with the desired length, duplicating values if necessary.

    Args:
        param: The parameter to be converted to a list.
        len_to_have: The desired length of the list.
        name: The name of the parameter for warning messages.

    Returns:
        List: The parameter as a list with the desired length.
    """
    if not isinstance(param, list):
        param = [
            copy.deepcopy(param) for _ in range(len_to_have)
        ]  # deepcopy allows to create independen copys of objects
    elif len(param) < len_to_have:
        print(
            f"Warning: The provided list for {name} has fewer values than the desired layers: {str(len_to_have)}. Duplicating values to match the number of layers."
        )
        param = param + [
            copy.deepcopy(param[0]) for _ in range(len_to_have % len(param))
        ]
        print(f"Final {name}: {param}")
    return param


def validate_keys_in_dict(dictionary: Dict, required_keys: List[str]) -> None:
    """
    Check if a dictionary contains all required keys.

    Args:
        dictionary: The dictionary to be checked.
        required_keys: A list of keys that must be present in the dictionary.

    Raises:
        ValueError: If any required key is missing in the dictionary.
    """
    for key in required_keys:
        if key not in dictionary:
            raise ValueError(f"Key '{key}' is missing in the dictionary: {dictionary}")


# From nnunet implementation
def empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        from torch import mps

        mps.empty_cache()
    else:
        pass


# From nnunet implementation
def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            # Handle variable-length arrays by concatenating along axis=0, allowing for different lengths
            try:
                collated[k] = np.concatenate([o[k] for o in outputs], axis=0)
            except ValueError:
                # If there's a shape mismatch, collect them as a list of arrays
                collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(
                f"Cannot collate input of type {type(outputs[0][k])}. "
                f"Modify collate_outputs to add this functionality"
            )
    return collated


def flatten_nested_numbers(value):
    if isinstance(value, np.ndarray):
        if np.prod(value.shape) == 1:
            return np.squeeze(value)

    elif isinstance(value, list):
        if len(value) == 1:
            return flatten_nested_numbers(value[0])
    else:
        return value
