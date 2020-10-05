import json
import torch
from typing import List
from pathlib import Path
from collections import defaultdict, Counter


class Tokenizer:
    pad_token = '[PAD]'
    bos_token = '[BOS]'
    eos_token = '[EOS]'
    unk_token = '[UNK]'
    special_tokens = [pad_token, bos_token, eos_token, unk_token]
    remove_in_decode = {pad_token, bos_token, eos_token}


class WordTokenizer(Tokenizer):
    """
    Simple word tokenizer with same interface as Huggingface tokenizer.
    """

    def __init__(self, text_file: Path = None,
                 vocab_file: Path = None,
                 coarse_classification=False):
        if text_file and vocab_file:
            self.w2i, self.i2w, self.label2idx = self.train(text_file, coarse_classification)
            self.save(vocab_file)
        elif vocab_file:
            self.w2i, self.i2w, self.label2idx = self.load(vocab_file)
        else:
            raise ValueError("Either text_file or vocab_file must be passed.")

        self.pad_token_id = self.w2i[self.pad_token]
        self.bos_token_id = self.w2i[self.bos_token]
        self.eos_token_id = self.w2i[self.eos_token]
        self.unk_token_id = self.w2i[self.unk_token]

    @property
    def vocab_size(self):
        return len(self.w2i)

    def encode(self, x, add_special_tokens=True):
        """
        Turn a sentence into a list of tokens. if add_special_tokens is True,
        add a start and stop token.
        
        Args:
            x (str): sentence to tokenize.
            add_special_tokens (bool, optional): if True, add a bos and eos token. 
                Defaults to True.
        
        Returns:
            list: list of integers.
        """
        encoded = [self.w2i.get(w, self.unk_token_id) for w in x.split()]
        if add_special_tokens:
            encoded = [self.bos_token_id] + encoded + [self.eos_token_id]
        return encoded

    def decode(self, x, skip_special_tokens=True):
        """
        Turn a list or torch.Tensor back into a sentence.
        If skip_special_tokens is True, all tokens in self.remove_in_decode are removed.
        
        Args:
            x (Iterable): Iterable or torch.Tensor of tokens.
            skip_special_tokens (bool, optional): Remove special tokens (leave [UNK]). 
                Defaults to True.
        
        Returns:
            str: decoded sentence.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        decoded = [self.i2w[i] for i in x]
        if skip_special_tokens:
            decoded = [t for t in decoded if t not in self.remove_in_decode]
        return ' '.join(decoded)

    def train(self, text_file, coarse_classification, max_vocab_size=None):
        """
        Train this tokenizer on a list of sentences.
        Method, split sentences, aggragate word counts, make a word to index (w2i)
        and index to word (i2w) dictionary from the max_vocab_size most common words.
        
        Args:
            text_file (str): Text to train the tokenizer on.
            max_vocab_size (int, optional): If defined, only keep the max_vocab_size most common words in the vocabulary. 
                Defaults to None.
        
        Returns:
            tuple(dict,dict,dict): c2i, i2c, label2idx
        """
        # if max_vocab_size < len(self.special_tokens):
        #     raise ValueError("Minimum vocab size is {}.".format(self.special_tokens))

        with open(text_file, "r") as f:
            text = f.readlines()
            word_counts = Counter()
            labels = []
            for line in text:
                line = line.split()
                label_str, question_str = line[0], line[1:]
                if coarse_classification:
                    label_str = label_str.split(":")[0]
                word_counts.update(question_str)
                labels.append(label_str)

        # Make vocabularies, sorted alphabetically
        label2idx = {label: idx for idx, label in enumerate(set(labels))}
        w2i = defaultdict(lambda: len(w2i))
        i2w = dict()

        # Default tokens
        for t in self.special_tokens:
            i2w[w2i[t]] = t

        # Give each word a token
        if max_vocab_size:
            words = [w[0] for w in word_counts.most_common(max_vocab_size - len(self.special_tokens))]
        else:
            words = list(word_counts.keys())    
        for word in sorted(words):
            i2w[w2i[word]] = word

        return dict(w2i), i2w, label2idx

    def save(self, vocab_file: Path):
        """ Save the tokenizer state """
        with open(vocab_file, "w") as f:
            vocab = {
                "w2i": self.w2i,
                "i2w": self.i2w,
                "label2idx": self.label2idx
            }
            json.dump(vocab, f, sort_keys=True)

    def load(self, vocab_file: Path):
        """ Load the tokenizer state """
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        return vocab["w2i"], vocab["i2w"], vocab["label2idx"]


class CharacterTokenizer(Tokenizer):

    def __init__(self, text_file: Path = None,
                 vocab_file: Path = None,
                 coarse_classification=False):
        if text_file and vocab_file:
            self.c2i, self.i2c, self.label2idx = self.train(text_file, coarse_classification)
            self.save(vocab_file)
        elif vocab_file:
            self.c2i, self.i2c, self.label2idx = self.load(vocab_file)
        else:
            raise ValueError("Either text_file or vocab_file must be passed.")

        self.pad_token_id = self.c2i[self.pad_token]
        self.bos_token_id = self.c2i[self.bos_token]
        self.eos_token_id = self.c2i[self.eos_token]
        self.unk_token_id = self.c2i[self.unk_token]

    @property
    def vocab_size(self):
        return len(self.c2i)

    def encode(self, x, add_special_tokens=False):

        encoded = [[self.c2i.get(c, self.unk_token_id) for c in word] for word in x.split()]

        if add_special_tokens:
            encoded = [self.bos_token_id] + encoded + [self.eos_token_id]
        return encoded

    def decode(self, x, skip_special_tokens=True):

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        decoded = [[self.i2c[i] for i in word] for word in x]
        if skip_special_tokens:
            decoded = [[t for t in word if t not in self.remove_in_decode] for word in decoded]

        decoded = [''.join(word) for word in decoded]
        return ' '.join(decoded)

    def train_on_data(self, data):
        """
        Returns:
            tuple: c2i, i2c dicts
        """
        char_counts = Counter()
        for sentence in data:
            chars = [char for word in sentence.split() for char in word]
            char_counts.update(chars)

        # Make vocabularies, sorted alphabetically
        c2i = defaultdict(lambda: len(c2i))
        i2c = dict()

        # Default tokens
        for t in self.special_tokens:
            i2c[c2i[t]] = t

        # Give each char a token
        characters = list(char_counts.keys())
        for char in characters:
            i2c[c2i[char]] = char

        return dict(c2i), i2c

    def train(self, text_file, coarse_classification, max_vocab_size=None):
        """
        Train this tokenizer on a list of sentences.
        Method, split sentences, aggragate character counts, make a char to index (c2i)
        and index to char (i2c) dictionary from the max_vocab_size most common characters.
        
        Args:
            text_file (str): Text to train the tokenizer on.
            max_vocab_size (int, optional): If defined, only keep the max_vocab_size most common characters in the vocabulary. 
                Defaults to None.
        
        Returns:
            tuple(dict,dict,dict): c2i, i2c, label2idx
        """
        # if max_vocab_size < len(self.special_tokens):
        #     raise ValueError("Minimum vocab size is {}.".format(self.special_tokens))

        with open(text_file, "r") as f:
            text = f.readlines()
            char_counts = Counter()
            labels = []
            for line in text:
                line = line.split()
                label_str, question_str = line[0], line[1:]
                chars = [char for word in question_str for char in word]
                char_counts.update(chars)
                if coarse_classification:
                    label_str = label_str.split(":")[0]
                labels.append(label_str)

        # Make vocabularies, sorted alphabetically
        label2idx = {label: idx for idx, label in enumerate(set(labels))}
        c2i = defaultdict(lambda: len(c2i))
        i2c = dict()

        # Default tokens
        for t in self.special_tokens:
            i2c[c2i[t]] = t

        # Give each char a token
        characters = list(char_counts.keys())
        for char in characters:
            i2c[c2i[char]] = char

        return dict(c2i), i2c, label2idx

    def save(self, vocab_file: Path):
        """ Save the tokenizer state """
        with open(vocab_file, "w") as f:
            vocab = {
                "c2i": self.c2i,
                "i2c": self.i2c,
                "label2idx": self.label2idx
            }
            json.dump(vocab, f, sort_keys=True)

    def load(self, vocab_file: Path):
        """ Load the tokenizer state """
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        return vocab["c2i"], vocab["i2c"], vocab["label2idx"]



# test block
if __name__ == "__main__":
    sen = ['good morning lovely people']
    tokenizer = WordTokenizer(sen)

    a = tokenizer.encode(sen[0])
    b = tokenizer.decode(a, skip_special_tokens=False)
    print(a)
    print(b)
    print(sen)
    print('---')

    tokenizer2 = CharacterTokenizer(sen)
    xx = tokenizer2.encode(sen[0])
    yy = tokenizer2.decode(xx, skip_special_tokens=False)
    print(xx)
    print(yy)
