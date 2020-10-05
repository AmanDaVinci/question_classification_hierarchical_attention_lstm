#!/usr/bin/env python3

###############################################################################
# Main entrypoint to the Question Classification project
# Starts training or test given an experiment config
###############################################################################

import os
import hydra
from omegaconf import DictConfig
from question_classification.trainer import Trainer
from question_classification.unsupervised_trainer import UnsupervisedTrainer


@hydra.main(config_path="configs", config_name="defaults.yaml")
def main(config: DictConfig) -> None:
    """ Runs the trainer based on the given experiment configuration """

    if config.test:
        # TODO: clean up current working directory with test=true
        experiment_path = os.getcwd().replace("test=true,", "").replace("test=True,", "")
        if config.unsupervised:
            trainer = UnsupervisedTrainer(config, experiment_path)
        else:
            trainer = Trainer(config, experiment_path)
        summary, report = trainer.test()
        print(summary)
        print(report)
    else:
        experiment_path = os.getcwd()
        if config.unsupervised:
            trainer = UnsupervisedTrainer(config, experiment_path)
        else:
            trainer = Trainer(config, experiment_path)
        trainer.run()
        print("Launched training. Press CTRL+C to stop.")
        print(f"Logs available at {os.getcwd()}")


if __name__ == "__main__":
    main()
