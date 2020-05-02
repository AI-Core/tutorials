from torchtext.data import Field, TabularDataset, BucketIterator
import torch
import os
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Any


logging.basicConfig(level=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def dataset_construction_from_csv(
        path: str,
        train_dataset: str,
        valid_dataset: str,
        test_dataset: str,
        features: List[str],
        labels: List[str],
        to_ignores: List[str],
        min_vocab_freq: int = 1
) -> Tuple[TabularDataset, TabularDataset, TabularDataset, int]:

    TEXT = Field(sequential=True, tokenize='spacy', lower=True)
    LABEL = Field(sequential=False, use_vocab=False)
    fields = []
    fields.extend([(feature, TEXT) for feature in features])
    fields.extend([(to_ignore, None) for to_ignore in to_ignores])
    fields.extend([(label, LABEL) for label in labels])
    # NOTE: for some reason, if we add the LABEL before None it causes an error as it doesn't recognize the fields
    # correctly. This could be a bug, as order should be irrelevant. Need to investigate further

    train, valid, test = TabularDataset.splits(
        path=path,
        train=train_dataset,
        validation=valid_dataset,
        test=test_dataset,
        format='csv',
        skip_header=True,
        fields=fields
    )
    TEXT.build_vocab(train, min_freq=min_vocab_freq)  # vocabulary is only built on training set
    vocab_length = len(TEXT.vocab)

    return train, valid, test, vocab_length


def iterator_construction(
        train: TabularDataset,
        valid: TabularDataset,
        test: TabularDataset,
        feature: str,  # the column name of the input text data
        batch_sizes: Tuple[int, int, int] = (64, 64, 64),  # order: train, valid, test
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train, valid, test),
        batch_sizes=(batch_sizes[0], batch_sizes[1], batch_sizes[2]),
        device=device,
        sort_key=lambda x: len(getattr(x, feature)),
        sort_within_batch=True
    )

    return train_iter, valid_iter, test_iter


class LSTMClass(nn.Module):
    def __init__(
            self,
            vocab_length: int,
            hidden_dim: int,
            output: int,  # number of classes for multi-class, or number of labels fir binary classification
            emb_dim: int = 300,
            num_lstm_layers: int = 1,
            num_linear_layers: int = 1,  # must be at least 1, or else this code as it's structured will fail
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_length, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_lstm_layers)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_linear_layers)])
        self.sigm = nn.Sigmoid()
        self.predictor = nn.Linear(hidden_dim, output)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(seq)
        hdn, _ = self.lstm(emb)  # hidden state, cell state (we don't need cell state now)
        feature = hdn[-1, :, :]  # taking the last output from the LSTM (we're dealing with many-to-one problem)
        for layer in self.linear_layers:
            feature = layer(feature)
            feature = self.sigm(feature)

        preds = self.predictor(feature)
        # SHOULD WE ADD A TANH OR SOME NON-LINEARITY HERE?
        return preds


def training(
        model: nn.Module,
        epochs: int,
        optimizer: Any,  # any optimizer from the module torch.optim
        loss_fn: Any,  # any loss from the module torch.nn
        train_iterator: BucketIterator,
        valid_iterator: BucketIterator,
        train: TabularDataset,
        valid: TabularDataset,
        feature: str,  # list should only contain one
        output_type: str,  # either "multiclass" or "binary_labels"
        labels: List[str]  # labels should only contain one element if you chose 'multiclass'
        # (multiple labels only supported for binary classification)
) -> None:
    assert any([output_type == option for option in ['multi_class', 'binary_class']]), \
        "Please choose output_type as either 'multi_class' or 'binary_labels'."
    if output_type == 'multi_class':
        assert len(labels) == 1, "labels should only contain one element if you chose 'multi_class' (multiple labels " \
                                 "only supported for binary classification)"

    train_dataloader = list(train_iterator)
    valid_dataloader = list(valid_iterator)
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        model.train()  # turn on training mode
        for batch in tqdm(train_dataloader):
            x = getattr(batch, feature)
            y = torch.cat([getattr(batch, label).unsqueeze(1) for label in labels], dim=1).float()
            # we will concatenate y into a single tensor
            optimizer.zero_grad()
            predictions = model(x)
            if output_type == 'multi_class':
                loss = loss_fn(predictions, y[:, 0].long())
            elif output_type == 'binary_class':
                loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data * x.size(0)

        epoch_loss = running_loss / len(train)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        model.eval()  # turn on evaluation mode
        with torch.no_grad():
            for batch in valid_dataloader:
                x = getattr(batch, feature)
                y = torch.cat([getattr(batch, label).unsqueeze(1) for label in labels], dim=1).float()
                predictions = model(x)
                if output_type == 'multi_class':
                    loss = loss_fn(predictions, y[:, 0].long())
                elif output_type == 'binary_class':
                    loss = loss_fn(predictions, y)
                val_loss += loss.data * x.size(0)

        val_loss /= len(valid)
        logger.info('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))


if __name__ == '__main__':
    folder_movie_reviews = os.path.join(os.getenv('DATA_DIR'), 'movie_review_dataset')
    folder_toxic_comments = os.path.join(os.getenv('DATA_DIR'), 'toxic_comments_dataset')
    train_dataset = 'train_dataset.csv'
    valid_dataset = 'valid_dataset.csv'
    test_dataset = 'test_dataset.csv'

    ###################################################
    # do training an validation on movie review dataset
    train, valid, test, voc_size = dataset_construction_from_csv(
        folder_movie_reviews, train_dataset, valid_dataset, test_dataset, ['text'], ['numerical_labels'], ['labels']
    )
    train_iter, valid_iter, test_iter = iterator_construction(train, valid, test, feature='text')
    model = LSTMClass(vocab_length=voc_size, hidden_dim=500, output=3, num_linear_layers=2)
    # output: 3 possible outcomes: negative, neutral, positive
    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.CrossEntropyLoss()
    training(
        model,
        epochs=3,
        optimizer=opt,
        loss_fn=loss_func,
        train_iterator=train_iter,
        valid_iterator=valid_iter,
        train=train,
        valid=valid,
        feature='text',
        output_type='multi_class',
        labels=['numerical_labels']
    )

    ###################################################
    # do training an validation on toxic comments dataset
    train, valid, test, voc_size = dataset_construction_from_csv(
        folder_toxic_comments,
        train_dataset,
        valid_dataset,
        test_dataset,
        ['comment_text'],
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        ['id']
    )
    train_iter, valid_iter, test_iter = iterator_construction(
        train, valid, test, batch_sizes=(7, 7, 7), feature='comment_text'
    )
    model = LSTMClass(vocab_length=voc_size, hidden_dim=500, output=6, num_linear_layers=2)
    # output: 6 binary labels
    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.BCEWithLogitsLoss()
    training(
        model,
        epochs=3,
        optimizer=opt,
        loss_fn=loss_func,
        train_iterator=train_iter,
        valid_iterator=valid_iter,
        train=train,
        valid=valid,
        feature='comment_text',
        output_type='binary_class',
        labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    )
