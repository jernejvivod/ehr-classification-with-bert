from typing import Iterable, Union

import torch
import torch.nn.functional as nnf
from sklearn import metrics
from torch.utils.data import DataLoader

from classification_with_embeddings import torch_device
from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier
from classification_with_embeddings.evaluation.util import _fasttext_data_to_x_y, _fasttext_data_to_x_y_multiple, \
    write_evaluation_prediction_data_to_file
from classification_with_embeddings.evaluation.visualization import write_classification_report, plot_confusion_matrix, \
    plot_roc


def evaluate_embeddings_model(clf: AClassifier,
                              method: Union[str, Iterable[str]],
                              test_data_path: Union[str, Iterable[str]],
                              results_path: str) -> None:
    """Evaluate embedding-based classifier on test data and get classification report, confusion matrix, and ROC plot.

    :param clf: AClassifier instance that outputs the predicted probabilities or label for a sample
    (document in FastText format)
    :param method: embedding method(s) used
    :param test_data_path: path(s) to test data in fastText format
    :param results_path: path to directory in which to store the results
    """

    logger.info('Computing predicted labels.')

    method_name = method if isinstance(method, str) else ', '.join(method)

    test_sentences, y_true = _fasttext_data_to_x_y(test_data_path) \
        if isinstance(test_data_path, str) else _fasttext_data_to_x_y_multiple(test_data_path)

    y_pred = clf.predict(test_sentences)

    logger.info('Saving evaluation results.')

    # write classification report
    classification_report = metrics.classification_report(y_true, y_pred)
    write_classification_report(classification_report, results_path, method_name)

    # visualize confusion matrix
    plot_confusion_matrix(y_pred, y_true, clf.classes(), clf.classes(), results_path, method_name)

    y_proba = None
    if _will_evaluate_roc(clf):
        logger.info('Computing predicted probabilities for ROC plot.')
        y_proba = clf.predict_proba(test_sentences)

        logger.info('Saving ROC plot.')
        plot_roc(y_proba, y_true, clf.classes()[1], results_path, method_name)

    # write evaluation data to file
    write_evaluation_prediction_data_to_file(
        method,
        results_path,
        y_pred.tolist(),
        y_proba,
        y_true,
        clf.classes()
    )


def _will_evaluate_roc(clf) -> bool:
    return clf.supports_predict_proba() and len(clf.classes()) == 2


def evaluate_cnn_model(model: torch.nn.Module,
                       test_data_loader: DataLoader,
                       results_path: str,
                       unique_labels: list,
                       class_names: list):
    """Evaluate CNN-based classifier on test data and get classification report, confusion matrix, and ROC plot.

    :param model: model to evaluate
    :param test_data_loader: DataLoader for test data
    :param results_path: path to directory in which to store the results
    :param unique_labels: unique labels present in the dataset
    :param class_names: names associated with the labels (in same order as the values specified for unique_labels)
    """

    logger.info('Using device: %s', torch_device)

    # prepare model for evaluation
    model.to(torch_device)
    model.eval()
    model.torch_device = torch_device

    # allocate empty tensors for stacking values in batches
    predicted_proba = torch.empty((0, 2)).to(torch_device)
    y_true = torch.empty(0, dtype=torch.int64).to(torch_device)

    with torch.no_grad():
        for batch in test_data_loader:
            # get inputs and labels for next batch
            inputs, labels = batch

            # compute loss
            y_proba_nxt = nnf.softmax(model(inputs), dim=1)
            predicted_proba = torch.cat((predicted_proba, y_proba_nxt), dim=0)
            y_true = torch.cat((y_true, labels.to(torch_device)))

    # get predictions from probabilities
    y_pred = torch.argmax(predicted_proba, dim=1)

    logger.info('Saving evaluation results.')

    # write classification report
    classification_report = metrics.classification_report(y_true.tolist(), y_pred.tolist())
    write_classification_report(classification_report, results_path, 'CNN')

    # visualize confusion matrix
    plot_confusion_matrix(y_pred.tolist(), y_true.tolist(), unique_labels, class_names, results_path, 'CNN')

    predicted_proba_numpy = None
    if len(unique_labels) == 2:
        predicted_proba_numpy = predicted_proba.cpu().numpy()
        plot_roc(predicted_proba_numpy, y_true.tolist(), unique_labels[1], results_path, 'CNN')

    # write evaluation data to file
    write_evaluation_prediction_data_to_file(
        'CNN',
        results_path,
        y_pred.tolist(),
        predicted_proba_numpy,
        y_true,
        class_names
    )
