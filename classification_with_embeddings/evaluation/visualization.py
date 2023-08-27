import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from classification_with_embeddings.evaluation import logger


def write_classification_report(cr: str, dir_path: str, method: str) -> None:
    """Write classification report to file.

    :param cr: classification report to write to file
    :param dir_path: path to directory in which to save the file containing the classification report
    :param method: file name (embedding method used)
    """

    output_file_path = os.path.abspath(os.path.join(dir_path, method + '_cr.txt'))
    logger.info('Writing classification report to {0}'.format(output_file_path))
    with open(output_file_path, 'w') as f:
        f.write(cr)


def plot_confusion_matrix(predictions: Union[list, np.ndarray],
                          y_test: list,
                          labels: list,
                          class_names: list,
                          plot_path: str,
                          method: str,
                          xlabel_text: str = None,
                          ylabel_text: str = None
                          ) -> None:
    """Plot confusion matrix.

    :param predictions: predictions of the classifier
    :param y_test: ground truth values
    :param labels: unique labels
    :param class_names: names associated with the labels (in same order)
    :param plot_path: path to directory in which to store the plot
    :param method: plot file name (embedding method used)
    :param xlabel_text: custom text for the x label (use default if None)
    :param ylabel_text: custom text for the y label (use default if None)
    """

    output_file_path = os.path.abspath(os.path.join(plot_path, method + '_cm' + '.svg'))
    logger.info('Saving confusion matrix plot to {0}'.format(output_file_path))

    # Plot confusion matrix and save plot.
    np.set_printoptions(precision=2)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        labels=labels,
        display_labels=class_names,
        y_true=y_test,
        y_pred=predictions,
        normalize='true',
        xticks_rotation='vertical'
    )

    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.tight_layout()

    if xlabel_text:
        plt.xlabel(xlabel_text)
    if ylabel_text:
        plt.ylabel(ylabel_text)

    plt.savefig(output_file_path, format='svg', bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_roc(scores: np.ndarray,
             y_test: list,
             pos_label,
             plot_path: str,
             method: str,
             xlabel_text: str = None,
             ylabel_text: str = None):
    """Plot ROC curve and compute the AUC metric.

    :param scores: scores for classes (probabilities)
    :param y_test: ground truth values
    :param pos_label: positive label
    :param plot_path: path to directory in which to store the plot
    :param method: plot file name (embedding method used)
    :param xlabel_text: custom text for the x label (use default if None)
    :param ylabel_text: custom text for the y label (use default if None)
    """

    output_file_path = os.path.abspath(os.path.join(plot_path, method + '_roc' + '.svg'))
    logger.info('Saving ROC plot to {0}'.format(output_file_path))

    # get false positive rates, true positive rates and thresholds
    fpr, tpr, _ = metrics.roc_curve(y_test, scores[:, 1], pos_label=pos_label)

    # compute AUC
    roc_auc = metrics.roc_auc_score(y_test, scores[:, 1])

    # plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {0:2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel_text if xlabel_text else 'False Positive Rate')
    plt.ylabel(ylabel_text if ylabel_text else 'True Positive Rate')

    plt.legend(loc="lower right")
    plt.savefig(output_file_path, format='svg', bbox_inches='tight')
    plt.clf()
    plt.close()
