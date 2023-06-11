import os

import evaluate
import torch
from torch.utils.data import DataLoader

from ehr_classification_with_bert import device, logger


def evaluate_model(model, eval_dataloader: DataLoader, results_path: str = '.'):
    """Evaluate model on test data

    :param model: Model to evaluate
    :param eval_dataloader: DataLoader for training data
    :param results_path: Path to directory in which to store the results
    """

    metric = evaluate.load('accuracy')
    model.eval()

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in batch.items()})

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])

    result = metric.compute()

    results_file = 'results.txt'

    logger.info('Saving results to %s', os.path.join(os.path.abspath(results_path), results_file))

    with open(os.path.abspath(os.path.join(results_path, results_file)), 'w') as f:
        f.write('accuracy: {:.2f}'.format(result['accuracy']))
