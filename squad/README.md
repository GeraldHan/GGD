## Setup
This repo is modified from [this link](https://github.com/chrischute/squad).

1. Downloads SQuAD 1.1 training and dev sets from https://rajpurkar.github.io/SQuAD-explorer/, as well as the GloVe 300-dimensional word vectors (840B) from http://nlp.stanford.edu/data_/glove.840B.300d.zip. The adversarial SQuAD data is provided in the `data` folder.

2. Run `python setup.py`
    This also pre-processes the dataset for efficient data loading with `setup_adv.py`. Modify the `parser.add_argument` for specific setups.

2. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code.
    Run the baseline and gge based on train.sh
