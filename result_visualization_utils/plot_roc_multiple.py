import argparse
import sys
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from result_visualization_utils import util


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # parse arguments
    parser = argparse.ArgumentParser(prog='plot-roc-multiple')
    parser.add_argument('--evaluation-data-paths', type=str, required=True, nargs='+',
                        help='Path to evaluation data files as produced by classification-with-embeddings')
    parser.add_argument('--method-names', type=str, required=False, nargs='*',
                        help='Names of evaluated methods (in same order as corresponding evaluation data files)')
    parser.add_argument('--xlabel-text', type=str,
                        help='Custom text for the x label')
    parser.add_argument('--ylabel-text', type=str,
                        help='Custom text for the y label')

    parsed_args = vars(parser.parse_args(argv[1:]))

    # parse scores and true values
    scores, true_values = util.parse_scores_and_true_values(parsed_args['evaluation_data_paths'])

    # get or parse method names
    if parsed_args['method_names'] is not None:
        method_names = parsed_args['method_names']
    else:
        method_names = util.parse_method_name_for_paths(parsed_args['evaluation_data_paths'])

    plot_roc_multiple(
        scores=scores,
        y_test=true_values,
        pos_label=None,
        method_names=method_names,
        xlabel_text=parsed_args['xlabel_text'],
        ylabel_text=parsed_args['ylabel_text']
    )


def plot_roc_multiple(scores: List[np.ndarray],
                      y_test: List[list],
                      pos_label,
                      method_names: List[str],
                      xlabel_text: str = None,
                      ylabel_text: str = None):
    """Plot multiple ROC curves from specified data.

    :param scores: scores for classes (probabilities)
    :param y_test: ground truth values
    :param pos_label: positive label
    :param method_names: names of methods
    :param xlabel_text: custom text for the x label (use default if None)
    :param ylabel_text: custom text for the y label (use default if None)
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

    plt.xlabel(xlabel_text if xlabel_text else 'False Positive Rate')
    plt.ylabel(ylabel_text if ylabel_text else 'True Positive Rate')

    plt.legend(loc="lower right")
    plt.savefig('roc_multiple.svg', format='svg', bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == '__main__':
    main()
