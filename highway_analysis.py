#!/usr/bin/env python3

###############################################################################
# Main entrypoint to the Question Classification project
# Starts training or test given an experiment config
###############################################################################

import os
import hydra
from omegaconf import DictConfig
from question_classification.trainer import Trainer


@hydra.main(config_path="configs", config_name="defaults.yaml")
def main(config: DictConfig) -> None:
    """ Runs the trainer based on the given experiment configuration """
    experiment_path = os.getcwd().replace("test=true,", "").replace("test=True,", "")
    trainer = Trainer(config, experiment_path)
    trainer.analyse_highway()


if __name__ == "__main__":
    main()
