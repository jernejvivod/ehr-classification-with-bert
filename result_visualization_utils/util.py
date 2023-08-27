import ast
from typing import Tuple, List

import numpy as np


def parse_predictions_and_true_values_for_paths(evaluation_data_paths: List[str]) -> Tuple[List[list], List[list]]:
    """Parse predicted and true values from evaluation data files produced by classification-with-embeddings.

    :param evaluation_data_paths: paths to evaluation data files
    """
    y_pred_list = []
    y_true_list = []
    for path in evaluation_data_paths:
        y_pred, y_true = parse_predictions_and_true_values_for_path(path)
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)
    return y_pred_list, y_true_list


def parse_predictions_and_true_values_for_path(evaluation_data_path: str) -> Tuple[list, list]:
    """Parse predicted and true values from evaluation data file produced by classification-with-embeddings.

    :param evaluation_data_path: path to evaluation data file
    """

    def parse_predictions_from_line(line: str) -> list:
        return ast.literal_eval(line[len('y_pred: '):])

    def parse_y_true_from_line(line: str) -> list:
        return ast.literal_eval(line[len('y_true: '):])

    with open(evaluation_data_path, 'r') as f:
        for line in f:
            if line.startswith('y_pred'):
                y_pred = parse_predictions_from_line(line)
            if line.startswith('y_true'):
                y_true = parse_y_true_from_line(line)

        if y_pred is None:
            raise ValueError('Predictions (predictions) not found in file.')
        if y_true is None:
            raise ValueError('True values (y_true) not found in file.')

        return y_pred, y_true


def parse_scores_and_true_values(evaluation_data_paths: List[str]) -> Tuple[List[np.ndarray], List[list]]:
    """Parse scores and true values from evaluation data files produced by classification-with-embeddings.

    :param evaluation_data_paths: paths to evaluation data files
    """
    scores_res = []
    true_values_res = []
    for path in evaluation_data_paths:
        score, true_values = parse_score_and_true_values(path)
        scores_res.append(score)
        true_values_res.append(true_values)
    return scores_res, true_values_res


def parse_score_and_true_values(evaluation_data_path: str) -> Tuple[np.ndarray, list]:
    """Parse scores and true values from evaluation data file produced by classification-with-embeddings.

    :param evaluation_data_path: path to evaluation data file
    """

    def parse_scores_from_line(line: str) -> np.ndarray:
        return np.array(ast.literal_eval(line[len('scores: '):].replace(' ', ',')))

    def parse_y_true_from_line(line: str) -> list:
        return ast.literal_eval(line[len('y_true: '):])

    with open(evaluation_data_path, 'r') as f:
        for line in f:
            if line.startswith('scores'):
                scores = parse_scores_from_line(line)
            if line.startswith('y_true'):
                y_true = parse_y_true_from_line(line)

        if scores is None:
            raise ValueError('Scores (scores) not found in file.')
        if y_true is None:
            raise ValueError('True values (y_true) not found in file.')

        return scores, y_true


def parse_labels_for_paths(evaluation_data_paths: List[str]) -> List[List[str]]:
    """Parse labels for evaluation data files produced by classification-with-embeddings.

    :param evaluation_data_paths: path to evaluation data files
    """

    labels_list = []
    for path in evaluation_data_paths:
        labels_list.append(parse_labels_for_path(path))
    return labels_list


def parse_labels_for_path(evaluation_data_path: str) -> List[str]:
    """Parse labels for evaluation data file produced by classification-with-embeddings.

    :param evaluation_data_path: path to evaluation data file
    """

    with open(evaluation_data_path, 'r') as f:
        for line in f:
            if line.startswith('labels'):
                return ast.literal_eval(line[len('labels: '):])


def parse_method_name_for_paths(evaluation_data_paths: List[str]):
    """Parse method names from evaluation data files produced by classification-with-embeddings.

    :param evaluation_data_paths: path to evaluation data files
    """

    method_list = []
    for path in evaluation_data_paths:
        method_list.append(parse_method_name_for_path(path))
    return method_list


def parse_method_name_for_path(evaluation_data_path: str) -> str:
    """Parse method names from evaluation data file produced by classification-with-embeddings.

    :param evaluation_data_path: path to evaluation data file
    """

    with open(evaluation_data_path, 'r') as f:
        for line in f:
            if line.startswith('method'):
                return line[len('method: '):-1]
    raise ValueError('File {} does not contain method name (method).'.format(evaluation_data_path))
