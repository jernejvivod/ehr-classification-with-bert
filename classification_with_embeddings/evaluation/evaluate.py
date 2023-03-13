from typing import Iterable

from sklearn import metrics

from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier
from classification_with_embeddings.evaluation.util import _fasttext_data_to_x_y, _fasttext_data_to_x_y_multiple
from classification_with_embeddings.evaluation.visualization import write_classification_report, plot_confusion_matrix, plot_roc


def evaluate(clf: AClassifier, method: str | Iterable[str], test_data_path: str | Iterable[str], results_path: str) -> None:
    """Evaluate embedding-based classifier on test data and get classification report, confusion matrix, and get ROC (AUC) plot.

    :param clf: AClassifier instance that outputs the predicted probabilities or label for a sample (fastText format document)
    :param method: embedding method used
    :param test_data_path: path to test data in fastText format
    :param results_path: path to directory in which to store the results
    """

    logger.info('Computing predicted labels.')

    method_name = method if isinstance(method, str) else ', '.join(method)

    test_sentences, y_true = _fasttext_data_to_x_y(test_data_path) if isinstance(test_data_path, str) else _fasttext_data_to_x_y_multiple(test_data_path)
    y_pred = clf.predict(test_sentences)

    logger.info('Saving evaluation results.')

    # write classification report
    classification_report = metrics.classification_report(y_true, y_pred)
    write_classification_report(classification_report, results_path, method_name)

    # visualize confusion matrix
    plot_confusion_matrix(y_pred, y_true, clf.classes(), clf.classes(), results_path, method_name)

    if _will_evaluate_roc(clf):
        logger.info('Computing predicted probabilities for ROC plot.')
        y_proba = clf.predict_proba(test_sentences)

        logger.info('Saving ROC plot.')
        plot_roc(y_proba, y_true, clf.classes()[1], results_path, method_name)


def _will_evaluate_roc(clf) -> bool:
    return clf.supports_predict_proba and len(clf.classes()) == 2
