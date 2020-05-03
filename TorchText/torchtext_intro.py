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
    """
    This function construct the train, validation and test datasets starting from raw .csv files. It also builds the
    vocabulary from the training dataset.
    :param path: the folder where the .csv files are stored.
    :param train_dataset: the raw .csv file with the training data.
    :param valid_dataset: the raw .csv file with the validation data.
    :param test_dataset: the raw .csv file with the testing data.
    :param features: list of strings, where each string represent the column name of each of the input features to the
           model, as they appear in the raw .csv file.
    :param labels: list of strings, where each string represent the column name of each of the output features (lables)
           of the model, as they appear in the raw .csv file.
    :param to_ignores: list of strings, where each string represent the column name of fields that appear in the raw
           .csv files which are neither features nor labels. These fields will be ignored.
    :param min_vocab_freq: the minimum frequency a word must have, in the training corpus, in order to be included in
           the vocabulary. Default: 1.
    :return: train: the training dataset, converted to a torchtest.data.TabularDataset
             valid: the evaluation dataset, converted to a torchtest.data.TabularDataset
             test: the testing dataset, converted to a torchtest.data.TabularDataset
             vocab_length: the size of the vocabulary built from the training dataset.
    """

    TEXT = Field(sequential=True, tokenize='spacy', lower=True)
    LABEL = Field(sequential=False, use_vocab=False)
    fields = []
    fields.extend([(feature, TEXT) for feature in features])
    fields.extend([(to_ignore, None) for to_ignore in to_ignores])
    fields.extend([(label, LABEL) for label in labels])

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
    """
    This function takes torchtext.data.TabularDataset's as input and output the correspondent BucketIterator's,
    splitting the datasets into batches. This iterator batches examples of similar lengths together, minimizing the
    amount of padding needed while producing freshly shuffled batches for each new epoch.
    :param train: a torchtest.data.TabularDataset representing the training dataset
    :param valid: a torchtest.data.TabularDataset representing the validation dataset
    :param test: a torchtest.data.TabularDataset representing the testing dataset
    :param feature: a string represent the name of the input feature to the model. Multiple inputs are not supported.
    :param batch_sizes: a tuple of 3 integers, each representing the batch size for the train, validation and test set
           respectively. Default: (64, 64, 64)
    :param device: the torch.device to be used, either 'cpu' or 'cuda' (gpu) if available
    :return: train_iter: a torchtext.data.BucketIterator, the iterator for the training dataset
             valid_iter: a torchtext.data.BucketIterator, the iterator for the validation dataset
             test_iter: a torchtext.data.BucketIterator, the iterator for the testing dataset
    """

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train, valid, test),
        batch_sizes=(batch_sizes[0], batch_sizes[1], batch_sizes[2]),
        device=device,
        sort_key=lambda x: len(getattr(x, feature)),
        sort_within_batch=True
    )

    return train_iter, valid_iter, test_iter


class LSTMClass(nn.Module):
    """
    This class implements a neural network with the following structure, suitable for a many-to-one regression or
    classification problem, such as sentiment analysis:
    - an initial embedding layer mapping numeric word tokens to an embedding matrix
    - a number of stacked LSTM layers (default 1 layer)
    - a number of linear fully-connected layers (default 1 layer), each followed by a sigmoid activation.
    NOTE: The output has not been normalised, as it is expected that the loss function of choice (defined in a
    separate training function) will apply the necessary scaling.
    """
    def __init__(
            self,
            vocab_length: int,
            hidden_dim: int,
            output: int,  # number of classes for multi-class, or number of labels fir binary classification
            emb_dim: int = 300,
            num_lstm_layers: int = 1,
            num_linear_layers: int = 1,  # must be at least 1, or else this code as it's structured will fail
    ) -> None:
        """
        :param vocab_length: the number of tokens (i.e. words) in our vocabulary
        :param hidden_dim: the size of the LSTM hidden layer, and the subsequent linear fully connected layers.
        :param output: for a multi-class model, this is the number of classes. For a binary classification problem, this
               is the number of labels (i.e. for a single binary label, it would be 1). Multiple labels are only
               supported for binary classification, not for multi-class classification.
        :param emb_dim: the size of the embedding layer.
        :param num_lstm_layers: the number of stacked LSTM layers. Default: 1.
        :param num_linear_layers: the number of stacked linear fully connected layers. Default: 1.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_length, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_lstm_layers)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_linear_layers)])
        self.tanh = nn.Tanh()
        self.predictor = nn.Linear(hidden_dim, output)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the model.
        :param seq: a torch.Tensor of shape(n, b), where n is sentence length, and b the batch size. Sentences shorter
               than n in the same batch have been padded. Each sentence is represented by the sequence of tokens it is
               composed of. Each token (word) is represented by an integer mapping that word in the vocabulary.
        :return: preds: the predictions of the model. A torch.Tensor of size (b, o) where b is batch size, and o is
                equal to the output parameter defined in the constructor.
        """
        emb = self.embedding(seq)
        all_hidden, (h_n, c_n) = self.lstm(emb)
        feature = h_n.squeeze()  # taking the last output from the LSTM (we're dealing with many-to-one problem)
        for layer in self.linear_layers:
            feature = layer(feature)
            feature = self.tanh(feature)

        preds = self.predictor(feature)
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
        output_type: str,  # either "multi_class" or "binary_labels"
        labels: List[str]  # labels should only contain one element if you chose 'multiclass'
        # (multiple labels only supported for binary classification)
) -> None:
    """
    This function implements the training of the LSTM model. It runs the model both in training and evaluation mode,
    and logs the losses on the console.
    :param model:
    :param epochs:
    :param optimizer: an appropriate optimizer from the module torch.optim
    :param loss_fn: an appropriate loss function from the module torch.nn
    :param train_iterator: the train iterator encoded as torchtext.data.BucketIterator
    :param valid_iterator: the validation iterator encoded as torchtext.data.BucketIterator
    :param train: the train dataset (already converted to a torchtext.data.TabularDataset)
    :param valid: the validation dataset (already converted to a torchtext.data.TabularDataset)
    :param feature: the string representing the column name of the textual input feature in the original .csv dataset
    :param output_type: a string, either "multi_class" for multi-class classification, or "binary_labels" for binary
           classification.
    :param labels: a list of string, where each string is one of the column names of the output labels in the original
           .csv dataset. If output_type is "multi_class", the labels list should only contain one string. Multiple
           labels are only supported for binary classification problems.
    :return: None.
    """
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
            if output_type == 'multi_class':
                y = getattr(batch, labels[0]).long()
            elif output_type == 'binary_class':
                y = torch.cat([getattr(batch, label).unsqueeze(1) for label in labels], dim=1).float()
            # we will concatenate y into a single tensor
            optimizer.zero_grad()
            predictions = model(x)
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
                    loss = loss_fn(predictions, y.long())
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
    train_iter, valid_iter, test_iter = iterator_construction(train, valid, test, feature='comment_text')
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
