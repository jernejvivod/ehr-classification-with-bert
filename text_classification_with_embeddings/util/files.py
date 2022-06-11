import os


def clear_file(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass
