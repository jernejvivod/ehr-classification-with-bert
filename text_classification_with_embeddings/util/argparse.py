import argparse
import os


def file_path(path: str) -> str:
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError('File \'{0}\' does not exist'.format(path))


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError('\'{0}\' is not a directory'.format(path))


def proportion_float(val: float) -> float:
    if val >= 0.0 and val <= 1.0:
        return val
    else:
        raise argparse.ArgumentTypeError('Value must be between 0.0 and 1.0')
