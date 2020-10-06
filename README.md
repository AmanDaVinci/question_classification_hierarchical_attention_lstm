# Question Classification using Attention-based Hierarchical LSTM

In this work, we investigate the application of attention-based hierarchical LSTM architecture variants to the problem of question classification. Question classification is often the first and important step for any Question-Answering system. We show that although the hierarchical design greatly improves performance over vanilla LSTMs, adding an attention mechanism only results in slight improvement. Then, we change perspective to probabilistically model the question dataset using discrete latent variables in order to see if the given coarse-level categories are re-discovered. While some latent structure is learned, it isn't the one we expected. We consider the possible reasons and suggest future improvements.

The following dataset is used in our experiments: [Experimental Data for Question Classification](http://cogcomp.org/Data/QA/QC/)
Our report is available at:

## How to run?
The training is based on the [hydra](https://hydra.cc/) config system for reproducibility. The data, model and training hyperparameters are specified as yaml files in the config directory. To train using the defaults on a GPU, just run ```python main.py training=on_gpu```.

To run a new training experiment which overrides the defaults, you can run a command as follows:  
```python main.py training=on_gpu seed=44 model=hier_attn_lstm training.tokenize_characters=true```

The results of an experiment is saved within a sub-directory in [experiments](experiments/) named after the overriden default parameters. This sub-directory contains the training logs, model checkpoints and tensorboard logs.

To evaluate the performance of the trained model on the testset just add the flag ```test=true``` to the original training command:
```python main.py test=true training=on_gpu seed=44 model=hier_attn_lstm training.tokenize_characters=true```

### Project Structure
The project is structured as follows:

```
question_classification\ [python package]
- models\
- datasets\
- trainer.py
- unsupervised_trainer.py
- tokenizer.py
data\
configs\
experiments\
notebooks\
main.py
```
The ```question_classification``` package consists of the model architectures, dataset classes, tokenizer and trainer. Based on the config, ```main.py``` runs an experiment using the ```question_classification``` package. The results are stored in a sub-directory inside ```experiments\```.

## References
1. Li, X. & Roth, D. Learning question classifiers. in Proceedings of the 19th international conference on Computational linguistics  - vol. 1 1–7 (Association for Computational Linguistics, 2002).
2. Xu, D. et al. Multi-class Hierarchical Question Classification for Multiple Choice Science Exams. arXiv:1908.05441 [cs] (2019).
