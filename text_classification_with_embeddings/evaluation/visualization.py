import os


# from . import logger


def write_classification_report(cr: str, dir_path: str, method: str) -> None:
    output_file_path = os.path.abspath(os.path.join(dir_path, method + '_cr.txt'))
    # logger.info('Writing classification report to {0}'.format(output_file_path))
    with open(output_file_path, 'w') as f:
        f.write(cr)
