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
