import os
import numpy as np

def get_last_checkpoint_path(checkpoint_dir):
    """Return path of the latest checkpoint in a given checkpoint directory."""
    paths = list(checkpoint_dir.glob('Epoch*'))
    if len(paths) > 0:
        # parse epochs and steps from path names
        epochs, steps = [], []
        for path in paths:
            epoch, step = path.stem.split('-')
            epoch = int(epoch.split('[')[-1][:-1])
            step = int(step.split('[')[-1][:-1])
            epochs.append(epoch)
            steps.append(step)
        # sort first by epoch, then by step
        last_model_ix = np.lexsort((steps, epochs))[-1]
        return paths[last_model_ix]