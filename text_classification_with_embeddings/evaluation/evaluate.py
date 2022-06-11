import numpy as np
from sklearn import metrics

from text_classification_with_embeddings import LABEL_WORD_PREFIX
from text_classification_with_embeddings.evaluation.visualization import write_classification_report, plot_confusion_matrix


def evaluate(clf, method: str, test_data_path: str, results_path: str) -> None:
    """evaluate embedding-based classifier on test data.

    :param clf: classifier function that outputs the predicted label for a sample (fastText format document)
    :param method: embedding method used
    :param test_data_path: path to test data in fastText format
    :param results_path: path to directory in which to store the results
    """

    # initialize lists for storing true and predicted class values
    y_true = []
    y_pred = []

    # go over test data and compute predicted labels
    with open(test_data_path, 'r') as f:
        for idx, sample in enumerate(f):

            # find ground-truth label
            label_search = [el for el in sample.split() if LABEL_WORD_PREFIX in el]
            if len(label_search) == 0:
                raise ValueError('Label not found in sample {0} in {1}'.format(idx, test_data_path))
            gt_label = label_search[0].replace(LABEL_WORD_PREFIX, '')
            y_true.append(gt_label)

            # get predicted label
            pred_label = clf(sample)
            y_pred.append(pred_label)

    # write classification report
    cr = metrics.classification_report(y_true, y_pred)
    write_classification_report(cr, results_path, method)

    # visualize confusion matrix
    labels_unique = np.unique(y_true)
    plot_confusion_matrix(y_pred, y_true, labels_unique, labels_unique, results_path, method)
