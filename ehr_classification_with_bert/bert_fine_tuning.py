from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

from ehr_classification_with_bert import logger, device


def fine_tune_bert(train_dataloader: DataLoader,
                   n_labels: int,
                   base_model: str = 'bert-base-cased',
                   model_save_path: str = '.',
                   n_epochs: int = 4) -> None:
    """Fine-tune a Hugging Face Transformers BERT model on a text classification task.

    :param train_dataloader: DataLoader for training data
    :param n_labels: Number of unique labels in the dataset
    :param base_model: Base BERT model to use
    :param model_save_path: Path to directory in which to save the fine-tuned model
    :param n_epochs: Number of training epochs to perform
    """

    logger.info('Fine-tuning BERT model with n_labels: %d, save_path: %s, num_epochs: %d',
                n_labels, model_save_path, n_epochs)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=n_labels
    )

    num_training_steps = n_epochs * len(train_dataloader)

    logger.info('Fine-tuning will take %d training steps.', num_training_steps)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)
    model.train()

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(n_epochs):
        for batch in train_dataloader:
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

    logger.info('Saving fine-tuned model to %s', model_save_path)
    model.save_pretrained(model_save_path)
