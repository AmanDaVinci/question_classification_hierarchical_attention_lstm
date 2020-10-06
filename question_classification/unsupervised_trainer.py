import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_
from omegaconf import DictConfig
from typing import Tuple

from question_classification.trainer import Trainer

BEST_MODEL_FNAME = "best-model.pt"


class UnsupervisedTrainer(Trainer):
    
    def __init__(self, config: DictConfig, experiment_path: str) -> None:
        super().__init__(config, experiment_path)
        pad_index = self.tokenizer.w2i[self.tokenizer.pad_token]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    
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
                self.writer.add_scalar('Train/Loss', results['loss'], self.current_iter)
                self.writer.add_scalar('Train/Perplexity', results['perplexity'], self.current_iter)
                report = (f"EPOCH:{epoch + 1} STEP:{i}/{num_batches}\t"
                          f"Accuracy: {results['accuracy']:.3f} "
                          f"Perplexity: {results['perplexity']:.3f} "
                          f"Loss: {results['loss']:.3f} "
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
        perplexities = []

        self.logger.debug("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
                perplexities.append(results['perplexity'])

        mean_accuracy = np.mean(accuracies)
        mean_perplexity = np.mean(perplexities)
        mean_loss = np.mean(losses)
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.save_checkpoint(BEST_MODEL_FNAME)

        self.writer.add_scalar('Valid/Accuracy', mean_accuracy, self.current_iter)
        self.writer.add_scalar('Valid/Loss', mean_loss, self.current_iter)
        self.writer.add_scalar('Valid/Perplexity', mean_perplexity, self.current_iter)
        summary = (f"[Validation]\t"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"Perplexity: {mean_perplexity:.3f} "
                  f"Total Loss: {mean_loss:.3f}")
        self.logger.info(summary)

    def test(self):
        """ Model testing and evaluation """

        print("Loading best model checkpoint... ")
        self.load_checkpoint(BEST_MODEL_FNAME)
        self.model.eval()
        losses = []
        accuracies = []
        perplexities = []

        print("Begin testing...")
        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
                perplexities.append(results['perplexity'])

        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        summary = (f"\n[Test Report]\n"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"Loss: {mean_loss:.3f}")
        return summary

    def _batch_iteration(self, batch: Tuple, training: bool):
        """ Iterate over one batch """
        x = batch[0].to(self.config.training.device)

        if training:
            self.model.train()
            self.opt.zero_grad()
            logits = self.model(x)
            logits = logits[1:].view(-1, logits.shape[-1])
            y = x[1:].view(-1)
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
                logits = logits[1:].view(-1, logits.shape[-1])
                y = x[1:].view(-1)
                loss = self.loss_fn(logits, y)

        y_pred = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        perplexity = torch.exp(loss)
        results = {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
            "accuracy": accuracy_score(y, y_pred),
        }
        return results
