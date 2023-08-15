import argparse
import ast
import sys
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # parse arguments
    parser = argparse.ArgumentParser(prog='train-test-set-stats')
    parser.add_argument('--evaluation-data-paths', type=str, required=True, nargs='+',
                        help='Path to evaluation data files as produced by classification-with-embeddings')
    parser.add_argument('--method-names', type=str, required=False, nargs='*',
                        help='Names of evaluated methods (in same order as corresponding evaluation data files)')

    parsed_args = vars(parser.parse_args(argv[1:]))

    # parse scores and true values
    scores, true_values = parse_scores_and_true_values(parsed_args['evaluation_data_paths'])

    # get or parse method names
    if parsed_args['method_names'] is not None:
        method_names = parsed_args['method_names']
    else:
        method_names = parse_method_names(parsed_args['evaluation_data_paths'])

    plot_roc_multiple(scores=scores, y_test=true_values, pos_label=None, method_names=method_names)


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


def parse_method_names(evaluation_data_paths: List[str]):
    """Parse method names from evaluation data file produced by classification-with-embeddings.

    :param evaluation_data_paths: path to evaluation data file
    """

    methods = []
    for path in evaluation_data_paths:
        method_nxt = None
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('method'):
                    method_nxt = line[len('method: '):-1]
        if method_nxt is None:
            raise ValueError('File {} does not contain method name (method).'.format(path))
        methods.append(method_nxt)
    return methods


def plot_roc_multiple(scores: List[np.ndarray], y_test: List[list], pos_label, method_names: List[str]):
    """Plot multiple ROC curves from specified data.

    :param scores: scores for classes (probabilities)
    :param y_test: ground truth values
    :param pos_label: positive label
    :param method_names: names of methods
    """

    # plot ROC curve
    plt.figure()
    lw = 2

    roc_curve_values = [metrics.roc_curve(y_test, scores[:, 1], pos_label=pos_label) for y_test, scores in zip(y_test, scores)]
    for (fpr, tpr, _), method_name in zip(roc_curve_values, method_names):
        plt.plot(fpr, tpr, lw=lw, label=method_name)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")
    plt.savefig('roc_multiple.svg', format='svg', bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == '__main__':
    main()
