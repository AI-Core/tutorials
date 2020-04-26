from torchtext.data import Field, TabularDataset, Iterator, BucketIterator
import torch
import os
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Tuple, Iterator as typ_Iterator, Union, Any


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
        to_ignores: List[str]
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
    TEXT.build_vocab(train)  # no need to build vocab for valid and test sets
    vocab_length = len(TEXT.vocab)

    return train, valid, test, vocab_length


def iterator_construction(
        train: TabularDataset,
        valid: TabularDataset,
        test: TabularDataset,
        batch_sizes: Tuple[int] = (64, 64, 64),  # order: train, valid, test
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[BucketIterator, BucketIterator, Iterator]:

    train_iter, valid_iter = BucketIterator.splits(
        (train, valid),
        batch_sizes=(batch_sizes[0], batch_sizes[1]),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True
    )

    test_iter = Iterator(
        test, batch_size=batch_sizes[2], device=device, sort=False, sort_within_batch=False
    )

    return train_iter, valid_iter, test_iter


class BatchWrapper:
    def __init__(self, iterator: Union[Iterator, BucketIterator], feature: str, label: Optional[List[str]]) -> None:
        self.iterator = iterator
        self.feature = feature
        self.label = label

    def __iter__(self) -> typ_Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.iterator:
            x = getattr(batch, self.feature)  # we assume only one input in this wrapper
            if self.label is not None:  # we will concatenate y into a single tensor
                y = torch.cat(
                    [getattr(batch, lab).unsqueeze(1) for lab in self.label], dim=1
                    # this is just in case we have multiple labels
                ).float()
            else:
                y = torch.zeros(1)
            yield (x, y)

    def __len__(self) -> int:
        # This is so the iteration will occur over the indices from zero to len-1.
        return len(self.iterator)


class SimpleLSTMBaseline(nn.Module):
    def __init__(
            self,
            vocab_length: int,
            hidden_dim: int,
            emb_dim: int = 300,
            num_lstm_layers: int = 1,
            num_linear_layers: int = 1  # must be at least 1, or else this code as it's structured will fail
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_length, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_lstm_layers)
        self.linear_layers = []
        for _ in range(num_linear_layers):  # - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
            preds = self.predictor(feature)
        return preds


def training(
        model: nn.Module,
        epochs: int,
        optimizer: Any,  # any optimizer from the module torch.optim
        loss_fn: Any,  # any loss from the module torch.nn
        train_iterator: BatchWrapper,
        valid_iterator: BatchWrapper,
        train: TabularDataset,
        valid: TabularDataset
) -> None:
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        model.train()  # turn on training mode
        for x, y in tqdm(train_iterator):
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(y, preds)
            loss.backward()
            optimizer.step()
            running_loss += loss.data * x.size(0)

        epoch_loss = running_loss / len(train)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        model.eval()  # turn on evaluation mode
        for x, y in valid_iterator:
            preds = model(x)
            loss = loss_fn(y, preds)
            val_loss += loss.data * x.size(0)

        val_loss /= len(valid)
        logger.info('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))


if __name__ == '__main__':
    folder = os.path.join(os.getenv('DATA_DIR'), 'movie_review_dataset')
    training_dataset = 'train_dataset.csv'
    validation_dataset = 'valid_dataset.csv'
    testing_dataset = 'test_dataset.csv'
    train, valid, test, voc_size = dataset_construction_from_csv(
        folder, training_dataset, validation_dataset, testing_dataset, ['text'], ['numerical_labels'], ['labels']
    )
    train_iter, valid_iter, test_iter = iterator_construction(train, valid, test)
    train_dataloader = BatchWrapper(train_iter, "text", ["numerical_labels"])
    valid_dataloader = BatchWrapper(valid_iter, "text", ["numerical_labels"])
    test_dataloader = BatchWrapper(test_iter, "text", None)  # no labels given for testing
    model = SimpleLSTMBaseline(vocab_length=voc_size, hidden_dim=500, emb_dim=100, num_linear_layers=3)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.BCEWithLogitsLoss()
    training(
        model,
        epochs=2,
        optimizer=opt,
        loss_fn=loss_func,
        train_iterator=train_dataloader,
        valid_iterator=valid_dataloader,
        train=train,
        valid=valid,
    )
