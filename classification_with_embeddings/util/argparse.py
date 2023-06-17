import argparse
import os


def file_path(path: str) -> str:
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError('File \'{0}\' does not exist.'.format(path))


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError('\'{0}\' is not a directory.'.format(path))


def proportion_float(val: str) -> float:
    try:
        val_float = float(val)
        if 0.0 <= val_float <= 1.0:
            return val_float
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('Value must be between 0.0 and 1.0')


def positive_int(value):
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("{} is not a positive integer.".format(value))
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("{} is not an integer.".format(value))
