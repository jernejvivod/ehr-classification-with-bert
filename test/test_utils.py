import os


def get_relative_path(file_path: str, path: str) -> str:
    return os.path.join(os.path.dirname(file_path), path)
