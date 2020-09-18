from question_classification.data.qc_dataset import QCDataset
from torch.utils.data import DataLoader
import os
from question_classification.tokenizers import WordTokenizer, CharacterTokenizer
from sklearn import preprocessing
import requests
import torch


# todo: probably need to add 'tokenizer' argument to init and add self.tokenizer
class QCDataLoader:
    """Dataloader for QC dataset found on https://cogcomp.seas.upenn.edu/Data/QA/QC/"""

    url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/'
    tokenizers = ['WordTokenizer', 'CharacterTokenizer']

    def __init__(self, tokenizer, filename='train_5500.label', split=(4362, 545, 545), batch_size=(64, 128, 128)):
        """
        :param tokenizer: str
        Name of the used tokenizer, choice: WordTokenizer, ...
        :param filename: str
        Name of the dataset
        :param split: tuple(int,int,int)
        train/dev/test splitting of the dataset
        :param batch_size: tuple(int,int,int)
        batch size of DataLoader for train/dev/test respectively
        """
        # tokenizer check
        if not tokenizer in self.tokenizers:
            raise ValueError('QCDataLoader: unknown tokenizer')

        # Make sure the datasets folder exists
        os.makedirs('./datasets', exist_ok=True)

        # Download dataset if it is not already present
        try:
            with open(f'datasets/{filename}', 'xb') as file:
                print(f'QCDataLoader: Downloading Dataset {filename}')
                response = requests.get(self.url + "/" + filename, allow_redirects=True)
                file.write(response.content)
        except FileExistsError:
            print(f'QCDataLoader: Dataset {filename} already present')

        # process data
        questions = []
        labels = []
        with open(f'datasets/{filename}', 'r', errors='replace') as data:
            for line in data.readlines():
                tokens = line.split()
                labels.append(tokens[0])
                questions.append(tokens[1:])

        self.max_sen_length = max([len(q) for q in questions])
        self.max_word_length = max([len(word) for sen in questions for word in sen])

        total = len(labels)
        if not sum(split) == total:
            raise ValueError(f'QCDataLoader: invalid splits; total of {total} exceeded')

        # get indices of the splits
        train = split[0]
        valid = split[0] + split[1]
        test = split[0] + split[1] + split[2]

        # fit label encoder
        self.le = preprocessing.LabelEncoder()
        self.le.fit(labels)
        enc_labels = self.le.transform(labels)

        # train tokenizer
        if tokenizer == 'WordTokenizer':
            q_str = [" ".join(q) for q in questions]
            self.tokenizer = WordTokenizer(q_str)
            enc_question = []
            for q in q_str:
                enc_question.append(self.tokenizer.encode(q, add_special_tokens=False))
            collate_fn = self._loader_collate_fn
        elif tokenizer == 'CharacterTokenizer':
            q_str = [" ".join(q) for q in questions]
            self.tokenizer = CharacterTokenizer(q_str)
            enc_question = []
            for q in q_str:
                enc_question.append(self.tokenizer.encode(q, add_special_tokens=False))
            collate_fn = self._create_char_collate_fn()
        else:
            raise NotImplementedError

        # make torch datasets
        self._train_data = QCDataset(enc_question[0:train], enc_labels[0:train])
        self._valid_data = QCDataset(enc_question[train:valid], enc_labels[train:valid])
        self._test_data = QCDataset(enc_question[valid:test], enc_labels[valid:test])

        # make Dataloaders
        self.train_loader = DataLoader(self._train_data, batch_size=batch_size[0], shuffle=True,
                                       collate_fn=collate_fn)
        self.valid_loader = DataLoader(self._valid_data, batch_size=batch_size[1], shuffle=False,
                                       collate_fn=collate_fn)
        self.test_loader = DataLoader(self._test_data, batch_size=batch_size[2], shuffle=False,
                                       collate_fn=collate_fn)

    @staticmethod
    def _loader_collate_fn(batch):
        """Collate function for DataLoader"""
        questions, labels = zip(*batch)
        lengths = [len(q) for q in questions]
        max_length = max(lengths)
        padded = [q + [0] * (max_length - len(q)) for q in questions]
        return torch.LongTensor(padded), torch.LongTensor(labels)

    def _create_char_collate_fn(self):
        def _char_collate_fn(batch):
            """Collate function for DataLoader"""
            questions, labels = zip(*batch)

            max_sen_length = self.max_sen_length
            max_word_length = self.max_word_length

            padded = [q + [[0]] * (max_sen_length - len(q)) for q in questions]
            padded = [[word + [0] * (max_word_length - len(word)) for word in sentence] for sentence in padded]
            return torch.LongTensor(padded), torch.LongTensor(labels)
        return _char_collate_fn


if __name__ == "__main__":
    DL = QCDataLoader('WordTokenizer')

    print(DL._train_data.__getitem__(5))
    print(DL.le.classes_)

    print('Train')
    for x,y in DL.train_loader:
        print(x)
        print(DL.tokenizer.decode(x[0]))
        print(y)
        print(type(x))
        break

    print('Valid')
    for x, y in DL.valid_loader:
        print(x)
        print(y)
        print(type(x))
        break

    print('Test')
    for x,y in DL.test_loader:
        print(x)
        print(y)
        print(type(x))
        break

    print('----')
    DL2 = QCDataLoader('CharacterTokenizer')

    for x,y in DL2.train_loader:
        print(x)
        print(DL2.tokenizer.decode(x[0]))
        print(y)
        print(type(x))
        print(x.shape)
        print(x[0])
        break
