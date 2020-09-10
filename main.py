#!/usr/bin/env python3

###############################################################################
# Main entrypoint to the Question Classification project
# Starts training or test given an experiment config
###############################################################################

import os
import hydra
from omegaconf import DictConfig
from question_classification.trainer import Trainer


@hydra.main(config_path="configs/defaults.yaml")
def main(config: DictConfig) -> None:
    """ Runs the trainer based on the given experiment configuration """

    if config.test:
        # TODO: clean up current working directory with inference=true
        experiment_path = os.getcwd().replace("inference=true,", "")
        trainer = Trainer(config, experiment_path)
        summary, report = trainer.test()
        print(summary)
        print(report)
    else:
        experiment_path = os.getcwd()
        trainer = Trainer(config, experiment_path)
        trainer.run()
        print("Launched training. Press CTRL+C to stop.")
        print(f"Logs available at {os.getcwd()}")


if __name__ == "__main__":
    main()
