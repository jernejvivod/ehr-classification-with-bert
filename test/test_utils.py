import os


def _get_relative_path(file_path: str, path: str):
    return os.path.join(os.path.dirname(file_path), path)