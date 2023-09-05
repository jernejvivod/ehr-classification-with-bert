import torch
import torch.nn as nn

from ehr_classification_with_bert import device


class EnsembleBertModel(nn.Module):
    def __init__(self, bert_model, emb_model, hidden_size=2048, freeze_bert_model=True, freeze_emb_model=True):
        """Ensemble model combining BERT and an embedding model that takes in text/documents and produces their
        embeddings. The BERT's last CLS token hidden state is concatenated with the feature vector produced by the
        embedding model. A series of fully-connected layers is then applied to perform classification.

        :param bert_model: Hugging Face BERT model to use in the ensemble
        :param emb_model: Embedding model to use in the ensemble
        :param hidden_size: Size of the hidden layers in the classifier
        :param freeze_emb_model: Freeze the layers of the BERT model (if using a PyTorch model)
        :param freeze_emb_model: Freeze the layers of the embedding model (if using a PyTorch model)
        """
        super().__init__()

        self.bert_model = bert_model
        self.emb_model = emb_model
        self.emb_model.device = 'cpu'

        self.emb_model_vector_size = self.emb_model([['test']]).shape[1]

        if freeze_bert_model:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        if isinstance(self.emb_model, nn.Module) and freeze_emb_model:
            for param in self.emb_model.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)

        # TODO study replacing with just a single linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.bert_model.config.hidden_size + self.emb_model_vector_size, hidden_size),
            self.relu,

            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            self.relu,

            nn.Linear(hidden_size, self.bert_model.config.num_labels)
        )

    def forward(self, **kwargs):
        # get text for related data and remove it from arguments
        # handle any nesting that may occur due to the use of a DataLoader with grouped data
        related_data_text = [ex[0] if isinstance(ex, tuple) else ex for ex in kwargs['related_data_1']]
        kwargs.pop('related_data_1')

        # get BERT CLS token embeddings
        bert_output = self.bert_model(**kwargs, output_hidden_states=True)
        bert_cls_embeddings = bert_output.hidden_states[-1][:, 0, :]

        # get embeddings model text embedding
        emb_model_embeddings = torch.Tensor(self.emb_model([ex.split(' ') for ex in related_data_text])).to(device)

        # get concatenated feature vector
        feature_vector = torch.cat((bert_cls_embeddings, emb_model_embeddings), dim=1)

        return self.classifier(feature_vector)
