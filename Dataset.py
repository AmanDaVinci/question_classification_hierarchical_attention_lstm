from torch.utils.data import Dataset


# todo: probably need to add 'tokenizer' argument to init and add self.tokenizer + return encoded question in __getitem__
class QCDataset(Dataset):
    """Torch Dataset"""
    def __init__(self, questions, labels):
        self.questions = questions
        self.labels = labels

    def __len__(self):
        """Returns the number of items in the dataset"""
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Returns the datapoint at index i as a tuple (sentence, label),
        where the sentence is tokenized.
        """
        return self.questions[idx], self.labels[idx]


if __name__ == "__main__":
    Data = QCDataset()
