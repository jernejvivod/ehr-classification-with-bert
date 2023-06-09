import os

import torch
import torch.nn.functional as nnf
from classification_with_embeddings.evaluation.visualization import write_classification_report, plot_confusion_matrix, \
    plot_roc
from evaluate import EvaluationModule
from sklearn import metrics
from torch.utils.data import DataLoader

from ehr_classification_with_bert import device, logger
from ehr_classification_with_bert.model.ensemble_bert_model import EnsembleBertModel


def evaluate_model(model, eval_dataloader: DataLoader, unique_labels, class_names, results_path: str = '.'):
    """Evaluate model on test data.

    :param model: Model to evaluate
    :param eval_dataloader: DataLoader for training data
    :param unique_labels: Unique labels present in the dataset
    :param class_names: Names associated with the labels (in same order as the values specified for unique_labels)
    :param results_path: Path to directory in which to store the results
    """

    logger.info('Evaluating model.')

    model.eval()

    # allocate empty tensors for stacking values in batches
    predicted_proba = torch.empty((0, 2)).to(device)
    y_true = torch.empty(0, dtype=torch.int64).to(device)

    for batch in eval_dataloader:

        # compute prediction
        with torch.no_grad():
            outputs = model(**{k: v.to(device) if (hasattr(v, 'to') and callable(getattr(v, 'to'))) else v
                               for k, v in batch.items()})

        # accumulate values
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        y_proba_nxt = nnf.softmax(logits, dim=1)
        predicted_proba = torch.cat((predicted_proba, y_proba_nxt), dim=0)
        y_true = torch.cat((y_true, batch['labels'].to(device)))

    # evaluate computed predictions and produce plots
    model_name = 'BERT' if not isinstance(model, EnsembleBertModel) else 'BERT_ENSEMBLE'
    evaluate_predictions(predicted_proba, y_true, unique_labels, class_names, model_name, results_path)


def evaluate_model_segmented(model, eval_dataloader: DataLoader, unique_labels, class_names, results_path: str = '.'):
    """Evaluate model on segmented test data by applying it to each segment and taking the mode of the segment
    predictions as the prediction for an example.

    The provided DataLoader must have a specified batch size of 1. It must provide tensors of shape [1, s, m] where s is
    the number of segments comprising an example. The first dimension is due to the batch size of 1 and is removed.

    The [s, m] tensors represent the segments and are fed to the model to get the segment predictions.

    :param model: Model to evaluate
    :param eval_dataloader: DataLoader for segmented training data
    :param unique_labels: Unique labels present in the dataset
    :param class_names: Names associated with the labels (in same order as the values specified for unique_labels)
    :param results_path: Path to directory in which to store the results
    """

    logger.info('Evaluating model on segmented data.')

    model.eval()

    # allocate empty tensors for stacking values in batches
    predicted_proba = torch.empty((0, 2)).to(device)
    y_true = torch.empty(0, dtype=torch.int64).to(device)

    for batch in eval_dataloader:

        # compute prediction
        with torch.no_grad():
            outputs = model(**{k: v[0].to(device) if (hasattr(v, 'to') and callable(getattr(v, 'to'))) else v
                               for k, v in batch.items()})

        # accumulate values
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        mean_logits_for_segments = torch.mean(logits, dim=0)  # compute mean logits for segments
        y_proba_nxt = nnf.softmax(mean_logits_for_segments, dim=0).unsqueeze(dim=0)
        predicted_proba = torch.cat((predicted_proba, y_proba_nxt), dim=0)
        y_true = torch.cat((y_true, batch['labels'][0][0].view(1).to(device)))

    # evaluate computed predictions and produce plots
    model_name = 'BERT' if not isinstance(model, EnsembleBertModel) else 'BERT_ENSEMBLE'
    evaluate_predictions(predicted_proba, y_true, unique_labels, class_names, model_name, results_path)


def evaluate_predictions(predicted_proba: torch.tensor,
                         y_true: torch.tensor,
                         unique_labels: list,
                         class_names: list,
                         model_name: str = '',
                         results_path: str = '.'):
    """Evaluate computed predictions on a test set.

    :param predicted_proba: Predicted probabilities of classes
    :param y_true: Ground truth values
    :param unique_labels: Unique labels present in the dataset
    :param class_names: Names associated with the labels (in same order)
    :param model_name: Name of the evaluated model (for formatting output filenames)
    :param results_path: Path to directory in which to store the results
    """

    # get predictions from probabilities
    y_pred = torch.argmax(predicted_proba, dim=1)
    logger.info('Saving evaluation results.')

    # write classification report
    classification_report = metrics.classification_report(y_true.tolist(), y_pred.tolist())
    write_classification_report(classification_report, results_path, model_name)

    # visualize confusion matrix
    plot_confusion_matrix(y_pred.tolist(), y_true.tolist(), unique_labels, class_names, results_path, model_name)
    if len(unique_labels) == 2:
        plot_roc(predicted_proba.numpy(), y_true.tolist(), unique_labels[1], results_path, model_name)
