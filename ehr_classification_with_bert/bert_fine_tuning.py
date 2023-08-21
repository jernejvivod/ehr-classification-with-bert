import os.path
from enum import Enum
from typing import List, Union

import torch
import torch.nn as nn
from classification_with_embeddings.embedding import embed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

from ehr_classification_with_bert import logger, device
from ehr_classification_with_bert.model.ensemble_bert_model import EnsembleBertModel


def fine_tune_bert(model_type: str,
                   train_dataloader: DataLoader,
                   val_dataloader: DataLoader,
                   n_labels: int,
                   eval_every_steps=500,
                   base_bert_model: str = 'bert-base-cased',
                   hidden_size: int = 32,
                   freeze_emb_model: bool = False,
                   model_save_path: str = '.',
                   n_epochs: int = 4,
                   train_data_path: Union[str, List[str]] = None,
                   emb_model_path: str = None,
                   emb_model_method: str = None,
                   emb_model_args: str = '',
                   **emb_model_kwargs) -> None:
    """Fine-tune a Hugging Face Transformers BERT model on a text classification task.

    :param model_type: Type of model to use. Valid values are 'bert' for a BERT-only model, 'ensemble-embeddings' for a
    BERT with aggregate embeddings ensemble, and 'ensemble-cnn' for a BERT with CNN-based embeddings ensemble
    :param train_dataloader: DataLoader for training data
    :param val_dataloader: DataLoader for validation data
    :param n_labels: Number of unique labels in the dataset
    :param eval_every_steps: Perform evaluation on validation data every specified number of steps
    :param base_bert_model: Base BERT model to use
    :param hidden_size: Size of the hidden layers in the classifier (if using ensemble model)
    :param freeze_emb_model: Freeze the ensembled model during fine-tuning
    :param model_save_path: Path to directory in which to save the fine-tuned model
    :param n_epochs: Number of training epochs to perform
    :param train_data_path: Path(s) to training data in FastText format
    :param emb_model_path: Path to stored model to use in an ensemble with BERT
    :param emb_model_method: Embedding method to use ('word2vec', 'fasttext', 'doc2vec', or 'starspace') if ensembling
    with an aggregate embeddings-based model
    :param emb_model_args: arguments passed to embedding implementation
    (key-value pairs such as val=1 enclosed in quotes with no commas separated by spaces)
    (required if ensembling with an aggregate embeddings-based model)
    :param emb_model_kwargs: Additional keyword arguments to pass to the ensembled aggregate embeddings-based model
    """

    logger.info('Fine-tuning model of type %s with n_labels: %d, save_path: %s, num_epochs: %d',
                model_type, n_labels, model_save_path, n_epochs)

    # load model
    model = get_model(
        model_type=model_type,
        base_bert_model=base_bert_model,
        n_labels=n_labels,
        hidden_size=hidden_size,
        freeze_emb_model=freeze_emb_model,
        emb_model_train_data_path=train_data_path if isinstance(train_data_path, str) else train_data_path[1],  # pass second dataset if multiple datasets used
        emb_model_path=emb_model_path,
        emb_model_method=emb_model_method,
        emb_model_args=emb_model_args,
        **emb_model_kwargs
    ).to(device)

    # compute number of required training steps
    num_training_steps = n_epochs * len(train_dataloader)

    logger.info('Fine-tuning will take %d training steps.', num_training_steps)

    # initialize optimizer ands learning rate sheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)
    lr_scheduler = get_scheduler(   
        name='linear', optimizer=optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps
    )

    # train model
    model.train()
    step_count = 0

    loss_fn = nn.CrossEntropyLoss()

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(n_epochs):
        for batch in train_dataloader:

            outputs = model(**{k: v.to(device) if (hasattr(v, 'to') and callable(getattr(v, 'to'))) else v
                               for k, v in batch.items()})
            loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs, batch['labels'].to(device))

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

            # validate_model model every N steps
            if val_dataloader and (step_count + 1) % eval_every_steps == 0:
                validate_model(model, batch)

            step_count += 1

    saved_model_path = os.path.join(model_save_path, 'trained_model.pth')

    logger.info('Saving fine-tuned model to %s', os.path.abspath(saved_model_path))
    torch.save(model, saved_model_path)


def validate_model(model: nn.Module, batch):
    print('\nPerforming validation on the training set.')

    # model will be evaluated
    model.eval()

    # initialize loss function
    loss_fn = nn.CrossEntropyLoss()


    with torch.no_grad():
        outputs = model(**{k: v.to(device) if (hasattr(v, 'to') and callable(getattr(v, 'to'))) else v
                            for k, v in batch.items()})
        loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs, batch['labels'].to(device))

        print("Validation Loss: {0:.4f}".format(loss))

    # model will continue to be trained
    model.train()


class ModelType(Enum):
    BERT_ONLY = 'bert'
    ENSEMBLE_EMBEDDINGS = 'ensemble-embeddings'
    ENSEMBLE_CNN = 'ensemble-cnn'


def get_model(model_type: str,
              base_bert_model: str,
              n_labels: int,
              hidden_size: int = 32,
              freeze_emb_model: bool = False,
              emb_model_train_data_path: str = None,
              emb_model_path: str = None,
              emb_model_method: str = 'word2vec',
              emb_model_args: str = '',
              **emb_model_kwargs):
    """Get model for fine-tuning.

    :param model_type: Model type to construct. Valid values are 'bert' for a BERT-only model, 'ensemble-embeddings' for
    a BERT with aggregate embeddings ensemble, and 'ensemble-cnn' for a BERT with CNN-based embeddings ensemble
    :param base_bert_model: Hugging Face BERT model to use
    :param n_labels: Number of unique labels in the dataset
    :param hidden_size: Size of the hidden layers in the classifier (if using ensemble model)
    :param freeze_emb_model: Freeze the ensembled model during fine-tuning
    :param emb_model_train_data_path: Path to training data in FastText format for the ensembled embeddings-based model
    :param emb_model_path: Path to stored model to use in an ensemble with BERT
    :param emb_model_method: Embedding method to use ('word2vec', 'fasttext', 'doc2vec', or 'starspace') if ensembling
    with an aggregate embeddings-based model
    :param emb_model_args: Arguments passed to embedding implementation
    (key-value pairs such as val=1 enclosed in quotes with no commas separated by spaces)
    :param emb_model_kwargs: Additional keyword arguments to pass to the ensembled aggregate embeddings-based model
    """

    # get BERT model
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        base_bert_model,
        num_labels=n_labels
    ).to(device)

    # get model instance to use
    if model_type == ModelType.BERT_ONLY.value:
        # BERT-ONLY MODEL
        return bert_model
    elif model_type == ModelType.ENSEMBLE_EMBEDDINGS.value:
        # BERT ENSEMBLED WITH AN AGGREGATE EMBEDDINGS-BASED MODEL
        emb_model = embed.get_doc_embedder_instance(
            method=emb_model_method,
            train_data_path=emb_model_train_data_path,
            method_args=emb_model_args,
            **emb_model_kwargs
        )
        return EnsembleBertModel(bert_model, emb_model, hidden_size, freeze_emb_model=freeze_emb_model)
    elif model_type == ModelType.ENSEMBLE_CNN.value:
        # BERT ENSEMBLED WITH A CNN-BASED MODEL
        cnn_emb_model = torch.load(emb_model_path, map_location=device)
        return EnsembleBertModel(
            bert_model,
            cnn_emb_model.feature_extractor,
            hidden_size,
            freeze_emb_model=freeze_emb_model
        )
    else:
        raise ValueError('Unsupported model type \'{}\''.format(model_type))
