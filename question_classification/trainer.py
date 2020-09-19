import os
import sys
import logging
import hydra
from omegaconf import DictConfig
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score, f1_score

from question_classification.data.qc_dataset import QCDataset
from question_classification.tokenizers import WordTokenizer, CharacterTokenizer

plt.style.use('seaborn-paper')
CHECKPOINTS = Path("model-checkpoints/")
LOGS = Path("tensorboard-logs/")
BEST_MODEL_FNAME = "best-model.pt"
DATA_DIR = Path("../../data/")


class Trainer():
    """ Trains an question classification model

    A trainer instantiates a model to be trained. It contains logic for training, validating, checkpointing, etc.
    All the parameters that drive the experiment behaviour are specified in a config dictionary.
    """

    def __init__(self, config: DictConfig, experiment_path: str) -> None:
        """ Instantiate a trainer for question classification models

        Parameters
        ----------
        config: DictConfig
            hydra config of parameters that drive the training behaviour
        experiment_path: str
            path to the current experiment directory

        """
        self.config = config

        # Experiment output directory
        self.exp_dir = Path(experiment_path)
        self.logger = logging.getLogger(__name__)

        # Checkpoint directory to save models        
        self.checkpoint_dir = self.exp_dir / CHECKPOINTS
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_exists = len(list(self.checkpoint_dir.glob('*'))) > 0

        # Tensorboard log directory
        self.log_dir = self.exp_dir / LOGS
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.logger.info("-"*50 + "\n" + "TRAINING LOG\n" + "-"*50)
        self.logger.info("-"*50 + '\n' + f'CONFIG:\n{config}\n' + "-"*50)

        if checkpoint_exists:
            self.logger.info(f"Checkpoint for {self.exp_dir.name} ALREADY EXISTS. Continuing training.")

        self.logger.info(f"Setting seed: {config.seed}")
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0.
        self.best_loss = float('inf')

        self.logger.info(f"Loading data {config.data.name} ...")
        data_dir = DATA_DIR / config.data.name
        token_level = "char" if config.training.tokenize_characters else "word"
        vocab_file = data_dir / f"vocab_{token_level}_{config.model.vocab_size}.json"
        train_file = data_dir / "train.txt"
        valid_file = data_dir / "valid.txt"
        test_file = data_dir / "test.txt"

        if not train_file.exists() or not valid_file.exists() or not test_file.exists():
            logging.info(f"Downloading and preparing data...")
            QCDataset.prepare(data_dir, config.training.tokenize_characters)

        if vocab_file.exists():
            logging.info(f"Loading trained tokenizer from {vocab_file}")
            self.tokenizer = CharacterTokenizer(vocab_file=vocab_file)\
                             if config.training.tokenize_characters else\
                             WordTokenizer(vocab_file=vocab_file)
        else:
            logging.info(f"Training tokenizer to {vocab_file}")
            self.tokenizer = CharacterTokenizer(train_file, vocab_file)\
                             if config.training.tokenize_characters else\
                             WordTokenizer(train_file, vocab_file)
            
        train_dataset = QCDataset(train_file, self.tokenizer) 
        valid_dataset = QCDataset(valid_file, self.tokenizer)
        test_dataset = QCDataset(test_file, self.tokenizer)
        collate_fn = QCDataset._char_collate_fn\
                     if config.training.tokenize_characters else\
                     QCDataset._word_collate_fn

        self.train_dl = DataLoader(train_dataset, batch_size=config.data.batch_size,
                                   shuffle=True, collate_fn=collate_fn)
        self.valid_dl = DataLoader(valid_dataset, batch_size=config.data.batch_size,
                                   shuffle=True, collate_fn=collate_fn)
        self.test_dl = DataLoader(test_dataset, batch_size=2*config.data.batch_size,
                                  shuffle=True, collate_fn=collate_fn)

        self.model = hydra.utils.instantiate(config.model)
        self.logger.info(f"Using device: {config.training.device}")
        self.model.to(config.training.device)
        self.opt = optim.Adam(self.model.parameters(), lr=config.training.lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.opt,
         lr_lambda=lambda epoch: config.training.lr_lambda ** epoch)
        self.loss_fn = nn.CrossEntropyLoss()

    def run(self):
        """Run the train-eval loop
        
        If the loop is interrupted manually, finalization will still be executed
        """
        try:
            self.logger.info(f"Begin training for {self.config.training.epochs} epochs")
            self.train()
        except KeyboardInterrupt:
            self.logger.warning("Manual interruption registered. Please wait to finalize...")
            self.finalize()

    def train(self):
        """ Main training loop """
        num_batches = len(self.train_dl)
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.logger.info(f'Current epoch: {self.current_epoch + 1} / {self.config.training.epochs}')
            self.current_epoch = epoch
            for i, batch in enumerate(self.train_dl):
                self.current_iter += 1
                t0 = time.time()
                results = self._batch_iteration(batch, training=True)
                time_spent = time.time() - t0
                rate = self.config.data.batch_size / time_spent

                self.writer.add_scalar('Train/Accuracy', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Train/F1-Score', results['f1_score'], self.current_iter)
                self.writer.add_scalar('Train/Loss', results['loss'], self.current_iter)
                report = (f"EPOCH:{epoch + 1} STEP:{i}/{num_batches}\t"
                          f"Accuracy: {results['accuracy']:.3f} "
                          f"F1-Score: {results['f1_score']:.3f} "
                          f"Speed: {rate :.1f} sentence/s ")
                self.logger.info(report)

                if i % self.config.training.valid_freq == 0:
                    self.validate()
                if i % self.config.training.save_freq == 0:
                    self.save_checkpoint()
            self.lr_scheduler.step()

    def validate(self):
        """ Main validation loop """
        self.model.eval()
        losses = []
        accuracies = []
        f1_scores = []
        Y, Y_pred = [], []

        self.logger.debug("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
                f1_scores.append(results['f1_score'])
                Y.append(results['y'])
                Y_pred.append(results['y_pred'])
            
        Y, Y_pred = np.concatenate(Y), np.concatenate(Y_pred)
        question_labels = list(self.tokenizer.label2idx.keys())
        report = classification_report(Y, Y_pred,
                                       labels=list(range(len(question_labels))), 
                                       target_names=question_labels,
                                       output_dict=True)
        # write classification report to tensorboard
        for label_name, numbers in report.items():
            if not isinstance(numbers, dict): break
            for metric, number in numbers.items():
                title = f"Valid/{label_name[1:].replace(':', '_')}/{metric.capitalize()}"
                self.writer.add_scalar(title, number, self.current_iter)
        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        mean_f1_score = np.mean(f1_scores)
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        self.writer.add_scalar('Valid/Accuracy', mean_accuracy, self.current_iter)
        self.writer.add_scalar('Valid/F1-Score', mean_f1_score, self.current_iter)
        self.writer.add_scalar('Valid/Loss', mean_loss, self.current_iter)
        summary = (f"[Validation]\t"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"F1-Score: {mean_f1_score:.3f} "
                  f"Total Loss: {mean_loss:.3f}")
        self.logger.info(summary)

    def test(self):
        """ Model testing and evaluation """

        print("Loading best model checkpoint... ")
        self.load_checkpoint(BEST_MODEL_FNAME)
        self.model.eval()
        losses = []
        accuracies = []
        f1_scores = []
        Y, Y_pred = [], []

        print("Begin testing...")
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
                f1_scores.append(results['f1_score'])
                Y.append(results['y'])
                Y_pred.append(results['y_pred'])

        Y, Y_pred = np.concatenate(Y), np.concatenate(Y_pred)
        question_labels = list(self.tokenizer.label2idx.keys())
        report = classification_report(Y, Y_pred, target_names=question_labels, output_dict=True)
        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        mean_f1_score = np.mean(f1_scores)
        summary = (f"\n[Test Report]\n"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"F1-Score: {mean_f1_score:.3f} "
                  f"Loss: {mean_loss:.3f}")
        return summary, report

    def _batch_iteration(self, batch: Tuple, training: bool):
        """ Iterate over one batch """
        x = batch[0].to(self.config.training.device)
        y = batch[1].to(self.config.training.device)
        
        if training:
            self.model.train()
            self.opt.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            if self.config.training.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(),
                                max_norm=self.config.training.max_grad_norm)
            self.opt.step()

        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

        y_pred = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        results = {
            "loss": loss.item(),
            "accuracy": accuracy_score(y, y_pred),
            "f1_score":  f1_score(y, y_pred, average='weighted'),
            "y": y,
            "y_pred": y_pred
        }
        return results

    def save_checkpoint(self, file_name: str = None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint directory and checkpoint file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = self.checkpoint_dir / file_name
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'model_state': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, file_name)
        self.logger.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, file_name: str):
        """Load the checkpoint with the given file name

        Checkpoint must contain:
            - current epoch
            - current iteration
            - model state
            - best accuracy achieved so far
            - optimizer state

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        try:
            file_name = self.checkpoint_dir / file_name
            self.logger.info(f"Loading checkpoint from {file_name}")
            checkpoint = torch.load(file_name, self.config.training.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['model_state'])
            self.opt.load_state_dict(checkpoint['optimizer'])

        except OSError:
            self.logger.error(f"No checkpoint exists @ {self.checkpoint_dir}")
        
    def finalize(self):
        """Finalize all necessary operations before stopping
        
        Saves checkpoint
        TODO: decide other finalization operations
        """
        self.save_checkpoint()

