import os
import requests
import torch
from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset

from question_classification.tokenizers import Tokenizer, WordTokenizer, CharacterTokenizer


class QCDataset(Dataset):
    """Question Classification Dataset"""

    url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"
    train_file = "train_5500.label"
    test_file = "TREC_10.label"

    def __init__(self, data_file: Path,
                 tokenizer: Tokenizer,
                 hierarchical_classification: bool = False,
                 coarse_classification: bool = False) -> None:
        self.questions, self.labels = [], []
        with open(data_file, "r", errors="replace") as data:
            text = data.readlines()
            for line in text:
                line = line.split()
                if hierarchical_classification:
                    # TODO: implement fine-level label classification here
                    pass
                else:
                    label_str, question_str = line[0], " ".join(line[1:])
                    if coarse_classification:
                        label_str = label_str.split(":")[0]
                question = tokenizer.encode(question_str, add_special_tokens=False)
                label = tokenizer.label2idx[label_str]
                self.questions.append(question)
                self.labels.append(label)

    def __len__(self) -> int:
        """Returns the number of items in the dataset"""
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[str, str]:
        """
        Returns the datapoint at index i as a tuple (sentence, label),
        where the sentence is tokenized.
        """
        return self.questions[idx], self.labels[idx]

    @staticmethod
    def _word_collate_fn(batch):
        """Word-level collate function for DataLoader"""
        questions, labels = zip(*batch)
        lengths = [len(q) for q in questions]
        max_length = max(lengths)
        padded = [q + [0] * (max_length - len(q)) for q in questions]
        return torch.LongTensor(padded), torch.LongTensor(labels)

    @staticmethod
    def _char_collate_fn(batch):
        """Character-level collate function for DataLoader"""
        questions, labels = zip(*batch)
        sen_lengths = [len(q) for q in questions]
        word_lengths = [len(word) for sen in questions for word in sen]
        max_sen_length = 37
        max_word_length = 28

        padded = [q + [[0]] * (max_sen_length - len(q)) for q in questions]
        padded = [[word + [0] * (max_word_length - len(word)) for word in sentence] for sentence in padded]
        return torch.LongTensor(padded), torch.LongTensor(labels)


    @classmethod
    def prepare(self, data_dir: Path,
                tokenize_characters: bool = False,
                train_valid_split: int = 4000) -> None:
        """ Download the data and prepare train, valid and test files

        Parameters
        ---
        data_dir: Path
            Path to the directory for storing question classification data.
        tokenize_characters: bool
            Whether to tokenize at character level. Defaults to word level.
        train_test_split: int
            Index where the train file will be split into train and validation set.
        """
        # create directory for storage
        data_dir.mkdir(parents=True)

        # download train data
        with open(data_dir/self.train_file, "xb") as file:
            response = requests.get(self.url+"/"+self.train_file,
                                    allow_redirects=True)
            file.write(response.content)

        # split into train and validation
        with open(data_dir/self.train_file, "r", errors="replace") as reader,\
             open(data_dir/"train.txt", "w") as train_writer,\
             open(data_dir/"valid.txt", "w") as valid_writer:
             data = reader.readlines()
             train_data = data[:train_valid_split]
             valid_data = data[train_valid_split:]
             train_writer.writelines(train_data)
             valid_writer.writelines(valid_data)
        os.remove(data_dir/self.train_file)

        # download test data
        with open(data_dir/"test.txt", "xb") as file:
            response = requests.get(self.url+"/"+self.test_file,
                                    allow_redirects=True)
            file.write(response.content)


if __name__ == "__main__":
    Data = QCDataset()
