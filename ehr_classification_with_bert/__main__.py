import argparse
import sys

import torch
from transformers import BertForSequenceClassification

from ehr_classification_with_bert import _util, Tasks, logger, device
from ehr_classification_with_bert._util import argparse_type_file_path, argparse_type_dir_path
from ehr_classification_with_bert.bert_evaluation import evaluate_model_segmented, evaluate_model
from ehr_classification_with_bert.bert_fine_tuning import fine_tune_bert, ModelType
from ehr_classification_with_bert.model.ensemble_bert_model import EnsembleBertModel


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

        # assert that correct number of data file paths specified for model type
        if parsed_args['model_type'] != ModelType.BERT_ONLY.value and \
                not isinstance(parsed_args['data_file_path'], list):
            raise ValueError('At least two data file paths should be specified when using model type \'{}\'.'
                             .format(parsed_args['model_type']))
        elif parsed_args['model_type'] == ModelType.BERT_ONLY.value and \
                not isinstance(parsed_args['data_file_path'], str):
            raise ValueError('Only one data file path should be specified when using model type \'{}\'.'
                             .format(parsed_args['model_type']))

        train_dataloader = _util.get_dataloader(
            data_file_path=parsed_args['data_file_path'],
            n_labels=parsed_args['n_labels'],
            batch_size=parsed_args['batch_size'],
            truncate_dataset_to=parsed_args['truncate_dataset_to'],
            split_above_tokens_limit=parsed_args['split_long_examples'],
            group_splits=False
        )

        val_dataloader = None
        if parsed_args['val_file_path'] is not None:
            val_dataloader = _util.get_dataloader(
                data_file_path=parsed_args['val_file_path'],
                n_labels=parsed_args['n_labels'],
                batch_size=parsed_args['batch_size'],
                truncate_dataset_to=None,
                split_above_tokens_limit=parsed_args['split_long_examples'],
                group_splits=False
            )

        fine_tune_bert(
            model_type=parsed_args['model_type'],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            n_labels=parsed_args['n_labels'],
            eval_every_steps=parsed_args['eval_every_steps'],
            step_lim=parsed_args['step_lim'],
            base_bert_model=parsed_args['base_bert_model'],
            hidden_size=parsed_args['hidden_size'],
            freeze_bert_model=parsed_args['freeze_bert_model'],
            freeze_emb_model=parsed_args['freeze_emb_model'],
            model_save_path=parsed_args['model_save_path'],
            n_epochs=parsed_args['n_epochs'],
            train_data_path=parsed_args['data_file_path'],
            bert_model_path=parsed_args['bert_model_path'],
            emb_model_path=parsed_args['emb_model_path'],
            emb_model_method=parsed_args['emb_model_method'],
            emb_model_args=parsed_args['emb_model_args'],
            starspace_path='./test'
        )

    elif parsed_args['task'] == Tasks.EVALUATE.value:
        logger.info('Running evaluate task.')

        loaded_model = torch.load(parsed_args['model_path'], map_location=device)

        # assert that correct number of data file paths specified for model type
        if isinstance(loaded_model, EnsembleBertModel) and not isinstance(parsed_args['data_file_path'], list):
            raise ValueError('At least two data file paths should be specified when using an ensemble model.')
        elif isinstance(loaded_model, BertForSequenceClassification) and not isinstance(parsed_args['data_file_path'], str):
            raise ValueError('Only one data file path should be specified when using a BERT model.')

        eval_dataloader = _util.get_dataloader(
            data_file_path=parsed_args['data_file_path'],
            n_labels=parsed_args['n_labels'],
            # batch size of 1 during testing is needed for the special classification procedure
            batch_size=1 if parsed_args['segmented'] else parsed_args['batch_size'],
            truncate_dataset_to=parsed_args['truncate_dataset_to'],
            split_above_tokens_limit=parsed_args['segmented'],
            group_splits=parsed_args['segmented']
        )

        if parsed_args['segmented']:
            evaluate_model_segmented(
                loaded_model,
                eval_dataloader,
                parsed_args['unique_labels'],
                parsed_args['class_names'],
                parsed_args['results_path']
            )
        else:
            evaluate_model(
                loaded_model,
                eval_dataloader,
                parsed_args['unique_labels'],
                parsed_args['class_names'],
                parsed_args['results_path']
            )


def _add_subparsers_for_fine_tune(subparsers):
    fine_tune_parser = subparsers.add_parser(Tasks.FINE_TUNE.value)
    fine_tune_parser.add_argument('--data-file-path', type=argparse_type_file_path, required=True, nargs='+',
                                  action=UnnestSingletonListElement,
                                  help='Path to file containing the fine-tuning data.')
    fine_tune_parser.add_argument('--val-file-path', type=argparse_type_file_path, nargs='+',
                                  action=UnnestSingletonListElement,
                                  help='Path to file containing the validation data.')
    fine_tune_parser.add_argument('--model-type', type=str, choices=[v.value for v in ModelType],
                                  default=ModelType.BERT_ONLY.value, help='Model type to use')
    fine_tune_parser.add_argument('--bert-model-path', type=_util.argparse_type_file_path,
                                  help='Path to stored BERT model to use')
    fine_tune_parser.add_argument('--emb-model-path', type=_util.argparse_type_file_path,
                                  help='Path to stored model to use in an ensemble with BERT.')
    fine_tune_parser.add_argument('--hidden-size', type=_util.argparse_type_positive_int, default=32,
                                  help='Size of hidden layers in the classifier used in the ensemble model.')
    fine_tune_parser.add_argument('--n-labels', type=_util.argparse_type_positive_int, required=True,
                                  help='Number of unique labels in the dataset.')
    fine_tune_parser.add_argument('--eval-every-steps', type=_util.argparse_type_positive_int,
                                  help='Perform evaluation on validation data every specified number of steps')
    fine_tune_parser.add_argument('--step-lim', type=_util.argparse_type_positive_int,
                                  help='Maximum number of training steps to perform.')
    fine_tune_parser.add_argument('--base-bert-model', type=str, default='bert-base-cased',
                                  help='Base BERT model to use.')
    fine_tune_parser.add_argument('--freeze-bert-model', action='store_true',
                                  help='Freeze the BERT model during fine-tuning')
    fine_tune_parser.add_argument('--freeze-emb-model', action='store_true',
                                  help='Freeze the ensembled model during fine-tuning')
    fine_tune_parser.add_argument('--model-save-path', type=argparse_type_dir_path, default='.',
                                  help='Path to directory in which to save the fine-tuned model.')
    fine_tune_parser.add_argument('--n-epochs', type=_util.argparse_type_positive_int, default=4,
                                  help='Number of epochs to use during fine-tuning.')
    fine_tune_parser.add_argument('--batch-size', type=_util.argparse_type_positive_int, default=32,
                                  help='Batch size to use during fine-tuning.')
    fine_tune_parser.add_argument('--truncate-dataset-to', type=_util.argparse_type_positive_int,
                                  help='Truncate the dataset to the specified number of samples.')
    fine_tune_parser.add_argument('--split-long-examples', action='store_true',
                                  help='Split examples whose token length is longer than the maximum length accepted by'
                                       ' the model into multiple examples.')
    fine_tune_parser.add_argument('--emb-model-method', type=str,
                                  choices=['word2vec', 'fasttext', 'doc2vec', 'starspace'], default='word2vec',
                                  help='Embedding method to use if ensembling with an aggregate embeddings-based model.')
    fine_tune_parser.add_argument('--emb-model-args', type=str, default='',
                                  help='Arguments passed to embedding model implementation (key-value pairs such as'
                                       ' val=1 enclosed in quotes with no commas separated by spaces)')


def _add_subparsers_for_evaluate(subparsers):
    fine_tune_parser = subparsers.add_parser(Tasks.EVALUATE.value)
    fine_tune_parser.add_argument('--model-path', type=argparse_type_file_path, required=True,
                                  help='Path to the saved model to evaluate.')
    fine_tune_parser.add_argument('--data-file-path', type=argparse_type_file_path, required=True, nargs='+',
                                  action=UnnestSingletonListElement,
                                  help='Path to file containing the evaluation data.')
    fine_tune_parser.add_argument('--n-labels', type=_util.argparse_type_positive_int, required=True,
                                  help='Number of unique labels in the dataset.')
    fine_tune_parser.add_argument('--batch-size', type=_util.argparse_type_positive_int, default=16,
                                  help='Batch size to use during evaluation. Ignored if --segmented flag is used.')
    fine_tune_parser.add_argument('--truncate-dataset-to', type=_util.argparse_type_positive_int,
                                  help='Truncate the dataset to the specified number of samples.')
    fine_tune_parser.add_argument('--segmented', action='store_true',
                                  help='Evaluate model on segmented test data by applying it to each segment and'
                                       ' taking the mode of the segment predictions as the prediction for an example.')
    fine_tune_parser.add_argument('--unique-labels', type=int, nargs='+', required=True,
                                  help='Unique labels present in the dataset')
    fine_tune_parser.add_argument('--class-names', type=str, nargs='+', required=True,
                                  help='Names associated with the labels (in same order)')
    fine_tune_parser.add_argument('--results-path', type=_util.argparse_type_dir_path, default='.',
                                  help='Path to directory in which to save the results')


class UnnestSingletonListElement(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
