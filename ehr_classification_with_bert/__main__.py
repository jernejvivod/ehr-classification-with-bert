import argparse
import sys

from datasets import Split
from transformers import AutoModelForSequenceClassification

from ehr_classification_with_bert import _util, Tasks, logger, device
from ehr_classification_with_bert._util import argparse_type_file_path, argparse_type_dir_path
from ehr_classification_with_bert.bert_evaluation import evaluate_model
from ehr_classification_with_bert.bert_fine_tuning import fine_tune_bert


def main(argv=None):
    if argv is None:
        argv = sys.argv

    logger.info('Using device: %s', device)

    # initialize argument parsers
    parser = argparse.ArgumentParser(prog='ehr-classification-with-bert')
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')
    _add_subparsers_for_fine_tune(subparsers)
    _add_subparsers_for_evaluate(subparsers)

    # run specified task task
    _run_task(vars(parser.parse_args(argv[1:])))


def _run_task(parsed_args: dict):
    if parsed_args['task'] == Tasks.FINE_TUNE.value:
        logger.info('Running fine-tune task.')

        train_dataloader = _util.get_dataloader(
            data_file_path=parsed_args['data_file_path'],
            n_labels=parsed_args['n_labels'],
            split=Split.TRAIN,
            batch_size=parsed_args['batch_size'],
            truncate_dataset_to=parsed_args['truncate_dataset_to']
        )

        fine_tune_bert(
            train_dataloader,
            n_labels=parsed_args['n_labels'],
            base_model=parsed_args['base_model'],
            model_save_path=parsed_args['model_save_path'],
            n_epochs=parsed_args['n_epochs']
        )

    elif parsed_args['task'] == Tasks.EVALUATE.value:
        logger.info('Running evaluate task.')

        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            parsed_args['model_path']
        ).to(device)

        eval_dataloader = _util.get_dataloader(
            data_file_path=parsed_args['data_file_path'],
            n_labels=parsed_args['n_labels'],
            split=Split.TEST,
            batch_size=parsed_args['batch_size'],
            truncate_dataset_to=parsed_args['truncate_dataset_to']
        )

        evaluate_model(loaded_model, eval_dataloader)


def _add_subparsers_for_fine_tune(subparsers):
    fine_tune_parser = subparsers.add_parser(Tasks.FINE_TUNE.value)
    fine_tune_parser.add_argument('--data-file-path', type=argparse_type_file_path, required=True,
                                  help='Path to file containing the fine-tuning data')
    fine_tune_parser.add_argument('--n-labels', type=_util.argparse_type_positive_int, required=True,
                                  help='Number of unique labels in the dataset')
    fine_tune_parser.add_argument('--base-model', type=str, default='bert-base-cased',
                                  help='Base BERT model to use')
    fine_tune_parser.add_argument('--model-save-path', type=argparse_type_dir_path, default='.',
                                  help='Path to directory in which to save the fine-tuned model')
    fine_tune_parser.add_argument('--n-epochs', type=_util.argparse_type_positive_int, default=4,
                                  help='Number of epochs to use during fine-tuning')
    fine_tune_parser.add_argument('--batch-size', type=_util.argparse_type_positive_int, default=16,
                                  help='Batch size to use during fine-tuning')
    fine_tune_parser.add_argument('--truncate-dataset-to', type=_util.argparse_type_positive_int,
                                  help='Truncate the dataset to the specified number of samples')


def _add_subparsers_for_evaluate(subparsers):
    fine_tune_parser = subparsers.add_parser(Tasks.EVALUATE.value)
    fine_tune_parser.add_argument('--model-path', type=argparse_type_dir_path, required=True,
                                  help='Path to directory containing the saved model to evaluate')
    fine_tune_parser.add_argument('--data-file-path', type=argparse_type_file_path, required=True,
                                  help='Path to file containing the evaluation data')
    fine_tune_parser.add_argument('--n-labels', type=_util.argparse_type_positive_int, required=True,
                                  help='Number of unique labels in the dataset')
    fine_tune_parser.add_argument('--batch-size', type=_util.argparse_type_positive_int, default=16,
                                  help='Batch size to use during evaluation')
    fine_tune_parser.add_argument('--truncate-dataset-to', type=_util.argparse_type_positive_int,
                                  help='Truncate the dataset to the specified number of samples')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
