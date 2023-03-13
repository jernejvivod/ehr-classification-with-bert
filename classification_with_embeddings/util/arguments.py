import json
from typing import Callable, Tuple, Iterable, List


def process_param_spec(params_str: str) -> dict:
    """Construction dictionary from function parameters specified using the command-line interface with key-value pairs (e.g. "val=3").

    :param params_str: string specifying the key value pairs
    :return: constructed dictionary
    """

    split_str = {kv[0]: kv[1] for exp in params_str.split() for kv in [exp.split('=')]}
    params_dict = dict()
    for k, v in split_str.items():
        try:
            params_dict[k] = int(v)
        except ValueError:
            try:
                params_dict[k] = float(v)
            except ValueError:
                params_dict[k] = v
    return params_dict


def get_train_and_val_paths_for_multiple_train_files(get_train_test_split_for_path_and_suffixes: Callable[[str, str, str], Tuple[str, str]], paths: Iterable[str], train_suffix: str, test_suffix: str) -> Tuple[List[str], List[str]]:
    """Get files corresponding to a train-test split of the data in specified files in fastText format where rows with the same index in the different files contain information about the same entity.

    :param get_train_test_split_for_path_and_suffixes: function that takes a path to a file containing training data, suffix to apply to file name containing the training data, and suffix to apply to file name containing the validation data
    :param paths: paths to files containing the samples in fastText format
    :param train_suffix: base suffix to apply to file name containing the training data (the index of the output file is appended to this suffix)
    :param test_suffix: base suffix to apply to file name containing the test data (the index of the output files is appended to this suffix)
    :return: tuple containing the paths to the output training files and paths to the output test files
    """

    train_paths = []
    val_paths = []
    for idx, path in enumerate(paths):
        train_path_nxt, test_path_nxt = get_train_test_split_for_path_and_suffixes(path, train_suffix + str(idx), test_suffix + str(idx))
        train_paths.append(train_path_nxt)
        val_paths.append(test_path_nxt)

    return train_paths, val_paths


def parse_param_grid(param_grid_path: str) -> dict:
    """Parse parameter grid for grid-search in JSON format to a dict.

    :param param_grid_path: path to file containing the parameter grid in JSON format
    :return: parsed parameter grid (dict)
    """
    with open(param_grid_path, 'r') as f:
        return json.loads(f.read())
