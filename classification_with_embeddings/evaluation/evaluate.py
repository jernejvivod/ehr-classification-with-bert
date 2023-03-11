import tqdm
from sklearn import metrics

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier
from classification_with_embeddings.evaluation.visualization import write_classification_report, plot_confusion_matrix, plot_roc


def evaluate(clf: AClassifier, method: str, test_data_path: str, results_path: str) -> None:
    """evaluate embedding-based classifier on test data.

    :param clf: AClassifier instance that outputs the predicted probabilities or label for a sample (fastText format document)
    :param method: embedding method used
    :param test_data_path: path to test data in fastText format
    :param results_path: path to directory in which to store the results
    """

    logger.info('Computing predicted labels.')

    sentences, y_true = _get_sentences_and_labels(test_data_path)
    y_pred = clf.predict(sentences)

    logger.info('Saving evaluation results.')

    # write classification report
    cr = metrics.classification_report(y_true, y_pred)
    write_classification_report(cr, results_path, method)

    # visualize confusion matrix
    plot_confusion_matrix(y_pred, y_true, clf.classes(), clf.classes(), results_path, method)

    if _will_evaluate_roc(clf):
        logger.info('Computing predicted probabilities for ROC plot.')
        y_proba = clf.predict_proba(sentences)

        logger.info('Saving ROC plot.')
        plot_roc(y_proba, y_true, clf.classes()[1], results_path, method)


def _get_sentences_and_labels(test_data_path: str):
    # initialize lists for storing samples and true class values
    x = []
    y_true = []

    with open(test_data_path, 'r') as f:
        for idx, sample in tqdm.tqdm(enumerate(f), desc='Computing predicted labels', unit='samples'):
            sample_split = sample.split()
            sample_split_no_label = [word for word in sample_split if LABEL_WORD_PREFIX not in word]
            x.append(sample_split_no_label)

            # find ground-truth label
            label_search = [el for el in sample_split if LABEL_WORD_PREFIX in el]
            if len(label_search) == 0:
                raise ValueError('Label not found in sample {0} in {1}'.format(idx, test_data_path))
            gt_label = label_search[0].replace(LABEL_WORD_PREFIX, '')
            y_true.append(gt_label)

    return x, y_true


def _will_evaluate_roc(clf) -> bool:
    return clf.supports_predict_proba and len(clf.classes()) == 2
