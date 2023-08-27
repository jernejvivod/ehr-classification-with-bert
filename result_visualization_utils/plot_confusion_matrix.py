import argparse
import os.path
import sys
from typing import List

from classification_with_embeddings.evaluation import visualization
from result_visualization_utils import util


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # parse arguments
    parser = argparse.ArgumentParser(prog='plot-confusion-matrix')
    parser.add_argument('--evaluation-data-paths', type=str, required=True, nargs='+',
                        help='Path to evaluation data files as produced by classification-with-embeddings')
    parser.add_argument('--class-names', type=str, required=True, nargs='+',
                        help='Names associated with the labels (in same order as specified in the evaluation data file)')
    parser.add_argument('--xlabel-text', type=str,
                        help='Custom text for the x label')
    parser.add_argument('--ylabel-text', type=str,
                        help='Custom text for the y label')

    parsed_args = vars(parser.parse_args(argv[1:]))

    evaluation_data_paths = parsed_args['evaluation_data_paths']

    # parse scores and true values
    y_pred_list, y_true_list = util.parse_predictions_and_true_values_for_paths(evaluation_data_paths)
    labels_list = util.parse_labels_for_paths(evaluation_data_paths)

    # get or parse method names
    method_list = util.parse_method_name_for_paths(evaluation_data_paths)

    class_names_proc = process_specified_class_names(parsed_args['class_names'], labels_list)

    for idx in range(len(evaluation_data_paths)):
        visualization.plot_confusion_matrix(
            y_pred_list[idx],
            y_true_list[idx],
            labels_list[idx],
            class_names_proc[idx],
            os.path.dirname(evaluation_data_paths[idx]),
            method_list[idx],
            xlabel_text=parsed_args['xlabel_text'],
            ylabel_text=parsed_args['ylabel_text']
        )


def process_specified_class_names(class_names: List[str], labels_list: List[List[str]]) -> List[List[str]]:
    """Group specified class names for specified evaluation data files.

    :param class_names: specified class names
    :param labels_list: list of lists of labels
    """

    class_names_res = []
    idx = 0
    for labels in labels_list:
        class_names_res.append(class_names[idx:len(labels)])
        idx += len(labels)
    return class_names_res


if __name__ == '__main__':
    main()
