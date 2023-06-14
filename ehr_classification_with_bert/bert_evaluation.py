import os

import evaluate
import torch
from evaluate import EvaluationModule
from torch.utils.data import DataLoader

from ehr_classification_with_bert import device, logger


def evaluate_model(model, eval_dataloader: DataLoader, results_path: str = '.'):
    """Evaluate model on test data.

    :param model: Model to evaluate
    :param eval_dataloader: DataLoader for training data
    :param results_path: Path to directory in which to store the results
    """

    logger.info('Evaluating model.')

    metric = evaluate.load('accuracy')
    model.eval()

    for batch in eval_dataloader:
        batch.pop('text')  # TODO should only remove if using BERT-only model

        # compute prediction
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in batch.items()})

        predictions = torch.argmax(outputs.logits, dim=-1)

        metric.add_batch(predictions=predictions, references=batch['labels'])

    write_results_to_file(metric, results_path=results_path)


def evaluate_model_segmented(model, eval_dataloader: DataLoader, results_path: str = '.'):
    """Evaluate model on segmented test data by applying it to each segment and taking the mode of the segment
    predictions as the prediction for an example.

    The provided DataLoader must have a specified batch size of 1. It must provide tensors of shape [1, s, m] where
    s is the number of segments comprising an example. The first dimension is due to the batch size of 1 and is removed.

    The [s, m] tensors represent the segments and are fed to the model to get the segment predictions.

    :param model: Model to evaluate
    :param eval_dataloader: DataLoader for segmented training data
    :param results_path: Path to directory in which to store the results
    """

    logger.info('Evaluating model on segmented data.')

    metric = evaluate.load('accuracy')
    model.eval()

    for batch in eval_dataloader:
        # unnest values in text column
        batch['text'] = [t[0] for t in batch['text']]
        batch.pop('text')  # TODO should only remove if using BERT-only model

        # compute prediction
        with torch.no_grad():
            outputs = model(**{k: v[0].to(device) for k, v in batch.items()})

        segment_predictions = torch.argmax(outputs.logits, dim=-1)
        predictions_mode = segment_predictions.mode().values

        metric.add_batch(predictions=predictions_mode.view(1), references=batch['labels'][0][0].view(1))

    write_results_to_file(metric, results_path=results_path)


def write_results_to_file(metric: EvaluationModule, results_path: str):
    """Write evaluation results to a file at the specified path.

    :param metric: EvaluationModule initialized with predictions
    :param results_path: Path to directory in which to store the results
    """

    result = metric.compute()

    results_file = 'results.txt'

    logger.info('Saving results to %s', os.path.join(os.path.abspath(results_path), results_file))

    with open(os.path.abspath(os.path.join(results_path, results_file)), 'w') as f:
        f.write('accuracy: {:.2f}'.format(result['accuracy']))
