import os


def delete_file(path) -> None:
    """delete file if exists

    :param path: path to file
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass
