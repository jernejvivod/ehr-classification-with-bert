import os.path
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_with_embeddings import torch_device
from classification_with_embeddings.embedding import logger
from classification_with_embeddings.embedding.cnn.dataset import FastTextFormatDataset, FastTextFormatCompositeDataset
from classification_with_embeddings.embedding.cnn.model.cnn_model import CnnTextClassificationModel
from classification_with_embeddings.embedding.cnn.model.multi_dataset_cnn_model import \
    CompositeCnnTextClassificationModel
from classification_with_embeddings.embedding.embed_util import get_word_to_embedding


def train_cnn_model(
        train_data_path: Union[str, List[str]],
        word_embeddings_path: str,
        n_labels: int,
        val_data_path: Optional[Union[str, List[str]]] = None,
        output_dir: str = '.',
        batch_size=32,
        n_epochs=3,
        max_filter_s=4,
        min_filter_s=2,
        filter_s_step=1,
        n_filter_channels=2,
        hidden_size=32,
        eval_every_steps=100
):
    """Train CNN-based document classification model.

    :param train_data_path: Path to file(s) containing the training data in FastText format
    :param word_embeddings_path: Path to file containing the word embeddings in TSV format
    :param n_labels: Number of unique labels in the dataset
    :param val_data_path: Path to file(s) containing the validation data in FastText format
    :param output_dir: Path to directory in which to save the trained model
    :param batch_size: Batch size to use during training
    :param n_epochs: Number of epochs to perform
    :param max_filter_s: Maximum filter bank height
    :param min_filter_s: Minimum filter bank height
    :param filter_s_step: Step in filter bank size
    :param n_filter_channels: Number of channels in filter bank
    :param hidden_size: Size of hidden layers in the classifier
    :param eval_every_steps: Perform evaluation on validation data every specified number of steps
    """

    logger.info('Using device: %s', torch_device)

    # initialize mapping of word to their embeddings
    word_to_embedding = {k: torch.tensor(v).float().to(torch_device) for k, v in
                         get_word_to_embedding(word_embeddings_path, False).items()}

    # initialize model
    if not (isinstance(train_data_path, str) or isinstance(train_data_path, list)):
        raise ValueError('Specified training data path(s) should be a str or a list.')
    model = CnnTextClassificationModel(
        word_to_embedding=word_to_embedding,
        n_labels=n_labels,
        max_filter_s=max_filter_s,
        min_filter_s=min_filter_s,
        filter_s_step=filter_s_step,
        n_filter_channels=n_filter_channels,
        hidden_size=hidden_size
    ).to(torch_device) if isinstance(train_data_path, str) else CompositeCnnTextClassificationModel(
        n_datasets=len(train_data_path),
        word_to_embedding=word_to_embedding,
        n_labels=n_labels,
        max_filter_s=max_filter_s,
        min_filter_s=min_filter_s,
        filter_s_step=filter_s_step,
        n_filter_channels=n_filter_channels,
        hidden_size=hidden_size
    ).to(torch_device)

    # initialize loss function, optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # initialize datasets and data loaders
    train_data_loader = get_dataloader(train_data_path, batch_size)

    val_data_loader = None
    if val_data_path:
        val_data_loader = get_dataloader(val_data_path, batch_size)

    # initialize progress bar
    num_training_steps = n_epochs * len(train_data_loader)
    progress_bar = tqdm(range(num_training_steps))

    # train model
    model.train()
    step_count = 0
    for epoch in range(n_epochs):
        for batch in train_data_loader:
            # get inputs and labels for next batch
            inputs, labels = batch
            labels = labels.to(torch_device)

            # compute loss
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # optimize
            loss.backward()
            optimizer.step()

            progress_bar.update(1)

            # validate_model model every N steps
            if val_data_loader and (step_count + 1) % eval_every_steps == 0:
                validate_model(model, val_data_loader)

            step_count += 1

        scheduler.step()

    saved_model_path = os.path.join(output_dir, 'trained_cnn_model.pth')
    logger.info('Saving trained model to %s', os.path.abspath(saved_model_path))
    torch.save(model, saved_model_path)


def validate_model(model: nn.Module, val_data_loader: DataLoader):
    print('\nPerforming validation on the training set.')

    # model will be evaluated
    model.eval()

    # initialize loss function
    loss_fn = nn.CrossEntropyLoss()

    # initialize accumulators
    validation_loss_acc = 0.0
    correct_acc = 0
    total_examples = 0

    with torch.no_grad():
        for batch in val_data_loader:
            # get inputs and labels for next batch
            inputs, labels = batch
            labels = labels.to(torch_device)

            # compute loss
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            validation_loss_acc += loss.item()

            # compute accuracy
            predictions = torch.argmax(logits, dim=1)
            total_examples += labels.size(0)
            correct_acc += predictions.eq(labels).sum().item()

    # compute average validation loss and accuracy
    average_loss = validation_loss_acc / len(val_data_loader)
    accuracy = correct_acc / total_examples

    print("Validation Loss: {0:.4f} | Accuracy: {1:.4f}".format(average_loss, accuracy))

    # model will continue to be trained
    model.train()


def get_dataloader(train_data_path: Union[str, List[str]], batch_size: int):
    if isinstance(train_data_path, str):
        dataset = FastTextFormatDataset(train_data_path)
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=FastTextFormatDataset.collate)
    elif isinstance(train_data_path, list):
        dataset = FastTextFormatCompositeDataset(train_data_path)
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=FastTextFormatCompositeDataset.collate)
    else:
        raise ValueError('Specified training data path(s) should be a str or a list.')

